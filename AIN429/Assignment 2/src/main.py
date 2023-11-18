from kafka import KafkaConsumer, KafkaProducer, TopicPartition, KStream

# Kafka Streams Setup
source_topic = 'input_topic'
target_topic = 'output_topic'

# Create Kafka Streams builder
streams_builder = KStream.builder()

# Define stream: Convert values to uppercase and send to target topic
streams_builder.stream(source_topic).map_values(lambda value: value.upper()).to(target_topic)

# Build and start the Kafka Streams application
streams = streams_builder.build()
streams.start()

# Kafka Consumer Configuration
consumer_config = {
    'bootstrap_servers': ['kafka-broker1:9092', 'kafka-broker2:9092'],
    'group_id': 'my-group',
    'auto_offset_reset': 'earliest',
    'value_deserializer': lambda x: x.decode('utf-8')
}

# Create Kafka Consumer instance
consumer = KafkaConsumer('my_topic', **consumer_config)

# Consume and process messages
for message in consumer:
    # Print message details: offset, key, and value
    print(f"offset: {message.offset}, key: {message.key}, value: {message.value}")

# Close the consumer
consumer.close()

# Kafka Producer Configuration
producer_config = {
    'bootstrap_servers': ['kafka-broker1:9092', 'kafka-broker2:9092'],
    'value_serializer': lambda v: str(v).encode('utf-8')
}

# Create Kafka Producer instance
producer = KafkaProducer(**producer_config)

# Publish a message to 'my_topic'
producer.send('my_topic', key='key', value='value')

# Close the producer
producer.close()
