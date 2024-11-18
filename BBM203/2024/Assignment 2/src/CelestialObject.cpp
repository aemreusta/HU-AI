#include "CelestialObject.h"

// Constructor to initialize CelestialObject with essential properties
CelestialObject::CelestialObject(const vector<vector<bool>> &shape, ObjectType type, int start_row,
                                 int time_of_appearance)
        : shape(shape), object_type(type), starting_row(start_row), time_of_appearance(time_of_appearance) {
    // No dynamic memory allocation in this constructor; pointers are initialized to nullptr by default
}

// Copy constructor for CelestialObject
CelestialObject::CelestialObject(const CelestialObject *other)
        : shape(other->shape),  // Copy the 2D vector shape
          object_type(other->object_type),  // Copy the object type
          starting_row(other->starting_row),  // Copy the starting row
          time_of_appearance(other->time_of_appearance)  // Copy the time of appearance
{
    // Deep copy pointers (initially nullptr since rotations are not passed/copied by constructor)
    right_rotation = nullptr;
    left_rotation = nullptr;
    next_celestial_object = nullptr;
}

// Function to delete rotations of a given celestial object. It should free the dynamically allocated
// memory for each rotation.
void CelestialObject::delete_rotations(CelestialObject *target) {
    if (!target) return;  // If the target is null, nothing to delete

    // Start from the target's right rotation and traverse the circular doubly-linked list
    CelestialObject *current = target->right_rotation;
    while (current && current != target) {
        CelestialObject *to_delete = current;
        current = current->right_rotation;

        // Break the circular link before deleting
        to_delete->right_rotation = nullptr;
        to_delete->left_rotation = nullptr;

        // Delete the current rotation
        delete to_delete;
    }

    // Finally, delete the target itself
    target->right_rotation = nullptr;
    target->left_rotation = nullptr;
}
