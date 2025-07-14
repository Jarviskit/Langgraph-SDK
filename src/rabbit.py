import asyncio
import aio_pika
import json
import ssl
from typing import Callable, final, Awaitable, cast
from .classes import MessageEvent

@final
class AsyncRabbitMQSubscriber:
    def __init__(
        self, 
        url: str,
        ssl_context: ssl.SSLContext | None = None,
        max_concurrent_workers: int = 1
    ):
        self.url = url
        self.ssl_context = ssl_context
        
        self.connection: aio_pika.RobustConnection | None = None
        self.channel: aio_pika.RobustChannel | None = None
        
        self.max_concurrent_workers = max_concurrent_workers
        self.semaphore = asyncio.Semaphore(max_concurrent_workers)
        self.active_tasks: set[asyncio.Task[None]] = set()

    async def connect(self) -> None:
        try:
            # Explicitly handle SSL configuration for clarity and debugging
            use_ssl = self.ssl_context is not None
            
            self.connection = await aio_pika.connect_robust(
                url=self.url,
                ssl_context=self.ssl_context,
                ssl=use_ssl,
                timeout=10  # Add timeout to prevent indefinite hangs
            )
            assert self.connection is not None  # Type narrowing for linter
            
            self.channel = await self.connection.channel()
            assert self.channel is not None  # Type narrowing for linter
            
            await self.channel.set_qos(prefetch_count=self.max_concurrent_workers)
            print("Connected to RabbitMQ")
            
        except aio_pika.exceptions.AMQPConnectionError as e:
            print(f"AMQP connection failed: {e}. Check if URL uses correct protocol/port (amqp://:5672 or amqps://:5671) and SSL matches server config.")
            raise
        except Exception as e:
            print(f"Failed to connect to RabbitMQ: {e}")
            print(f"Connection details: {self.url}")
            raise
    
    async def declare_queue(self, queue_name: str, durable: bool = True) -> None:
        if not self.channel:
            raise RuntimeError("Channel not initialized. Call connect() first.")
        
        await self.channel.declare_queue(queue_name, durable=durable)
    
    async def subscribe(
        self,
        queue_name: str,
        callback: Callable[[MessageEvent], Awaitable[bool]]
    ) -> None:
        if not self.channel:
            raise RuntimeError("Channel not initialized. Call connect() first.")

        queue = await self.channel.declare_queue(queue_name, durable=True)
        async def process_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
            async with self.semaphore:
                async with message.process(ignore_processed=True):
                    headers = message.headers or {}
                    retry_count = int(cast(str, headers.get("x-retry", 0)))

                    if retry_count > 2:
                        print("Max retries reached. Discarding message.")
                        await message.reject(requeue=False)
                        return

                    try:
                        body = message.body.decode("utf-8")
                        event = MessageEvent(json.loads(body))

                        success = await callback(event)

                        if success:
                            await message.ack()
                        else:
                            print("Message processing failed. Retrying...")
                            await message.ack()  # Acknowledge before re-publish
                            await asyncio.sleep(3)

                            new_headers = dict(headers)
                            new_headers["x-retry"] = retry_count + 1

                            if self.channel:
                                await self.channel.default_exchange.publish( # type: ignore
                                    aio_pika.Message(
                                        body=message.body,
                                        headers=new_headers,
                                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                                    ),
                                    routing_key=queue_name
                                )

                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        await message.reject(requeue=False)

                    except Exception as e:
                        print(f"Unhandled exception: {e}")
                        await message.reject(requeue=True)

        async def wrapper(message: aio_pika.abc.AbstractIncomingMessage) -> None:
            # Create a task for each message to enable concurrent processing
            task = asyncio.create_task(process_message(message))
            self.active_tasks.add(task)
            
            # Clean up completed tasks
            def cleanup_task(task: asyncio.Task[None]) -> None:
                self.active_tasks.discard(task)
            
            task.add_done_callback(cleanup_task)

        await queue.consume(wrapper)
        print(f"Subscribed to queue '{queue_name}' with {self.max_concurrent_workers} concurrent workers. Waiting for messages...")
        
        # Keep the consumer running indefinitely
        try:
            await asyncio.Future()  # Run forever
        except asyncio.CancelledError:
            print("Consumer cancelled")
        except KeyboardInterrupt:
            print("Consumer interrupted by user")
        finally:
            # Wait for all active tasks to complete
            if self.active_tasks:
                print(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
                await asyncio.gather(*self.active_tasks, return_exceptions=True)

    async def close(self) -> None:
        # Cancel all active tasks
        if self.active_tasks:
            print(f"Cancelling {len(self.active_tasks)} active tasks...")
            for task in self.active_tasks:
                task.cancel()
            
            # Wait for all tasks to be cancelled
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
            self.active_tasks.clear()

        if self.channel and not self.channel.is_closed:
            await self.channel.close()

        if self.connection and not self.connection.is_closed:
            await self.connection.close()

        print("RabbitMQ connection closed")
