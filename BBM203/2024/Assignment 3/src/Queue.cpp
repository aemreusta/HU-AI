#include "Queue.h"
#include <iostream>

// Constructor to initialize an empty queue
Queue::Queue() : front(0), rear(MAX_QUEUE_SIZE - 1), size(0) {}

// Adds a province to the end of the queue
void Queue::enqueue(int province) {
    if (size == MAX_QUEUE_SIZE) {
        std::cerr << "Error: Queue overflow" << std::endl;
        return;
    }
    rear = (rear + 1) % MAX_QUEUE_SIZE;
    data[rear] = province;
    size++;
}

// Removes and returns the front province from the queue
int Queue::dequeue() {
    if (isEmpty()) {
        std::cerr << "Error: Queue underflow" << std::endl;
        return -1; // Return an invalid value to indicate error
    }
    int province = data[front];
    front = (front + 1) % MAX_QUEUE_SIZE;
    size--;
    return province;
}

// Returns the front province without removing it
int Queue::peek() const {
    if (isEmpty()) {
        std::cerr << "Error: Queue is empty" << std::endl;
        return -1; // Return an invalid value to indicate error
    }
    return data[front];
}

// Checks if the queue is empty
bool Queue::isEmpty() const {
    return size == 0;
}

// Add a priority neighboring province in a way that will be dequeued and explored before other non-priority neighbors
void Queue::enqueuePriority(int province) {
    if (size == MAX_QUEUE_SIZE) {
        std::cerr << "Error: Queue overflow" << std::endl;
        return;
    }
    front = (front - 1 + MAX_QUEUE_SIZE) % MAX_QUEUE_SIZE;
    data[front] = province;
    size++;
}