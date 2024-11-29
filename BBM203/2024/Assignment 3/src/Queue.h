#ifndef QUEUE_H
#define QUEUE_H

#define MAX_QUEUE_SIZE 100  // Define a maximum size for the queue

class Queue {
private:
    int front, rear, size;   // Indices for the front and rear elements of the queue, and the current size
    int data[MAX_QUEUE_SIZE]; // Static array to store provinces as integers (province IDs)

public:
    Queue();                  // Constructor to initialize the queue

    // Adds a province to the end of the queue
    void enqueue(int province);

    // Removes and returns the front province from the queue
    int dequeue();

    // Returns the front province without removing it
    int peek() const;

    // Checks if the queue is empty
    bool isEmpty() const;

    // Add a priority province
    void enqueuePriority(int province);
};

#endif // QUEUE_H