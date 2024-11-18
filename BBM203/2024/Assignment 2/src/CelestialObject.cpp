#include "CelestialObject.h"
#include <iostream> // For debugging output

// Constructor to initialize CelestialObject with essential properties
CelestialObject::CelestialObject(const vector<vector<bool>> &shape, ObjectType type, int start_row, int time_of_appearance)
        : shape(shape), object_type(type), starting_row(start_row), time_of_appearance(time_of_appearance) {
    if (shape.empty()) {
        std::cerr << "Error: CelestialObject initialized with an empty shape!" << std::endl;
    } else {
        for (const auto &row : shape) {
            if (row.empty()) {
                std::cerr << "Error: CelestialObject contains an empty row!" << std::endl;
                break;
            }
        }
    }

    right_rotation = nullptr;
    left_rotation = nullptr;
    next_celestial_object = nullptr;
}


// Copy constructor for CelestialObject
CelestialObject::CelestialObject(const CelestialObject *other)
        : shape(other->shape),
          object_type(other->object_type),
          starting_row(other->starting_row),
          time_of_appearance(other->time_of_appearance) {
    if (other->shape.empty()) {
        std::cerr << "Warning: Copying CelestialObject with an empty shape!" << std::endl;
    }
    right_rotation = nullptr;
    left_rotation = nullptr;
    next_celestial_object = nullptr;
}

// Function to delete rotations of a given celestial object
void CelestialObject::delete_rotations(CelestialObject *target) {
    if (!target) return;

    CelestialObject *current = target->right_rotation;
    while (current && current != target) {
        CelestialObject *to_delete = current;
        current = current->right_rotation;
        delete to_delete;
    }

    target->right_rotation = nullptr;
    target->left_rotation = nullptr;
}

// Function to rotate a shape clockwise
vector<vector<bool>> CelestialObject::rotate_shape_right(const vector<vector<bool>> &shape) {
    if (shape.empty()) {
        std::cerr << "Error: Attempting to rotate an empty shape!" << std::endl;
        return {};
    }

    int rows = shape.size();
    int cols = shape[0].size();
    vector<vector<bool>> rotated_shape(cols, vector<bool>(rows, false));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            rotated_shape[j][rows - i - 1] = shape[i][j];
        }
    }

    return rotated_shape;
}

// Function to generate all rotations for a celestial object
void CelestialObject::generate_rotations() {
    if (shape.empty()) {
        std::cerr << "Error: Cannot generate rotations for an empty shape!" << std::endl;
        return;
    }

    for (const auto &row : shape) {
        if (row.empty()) {
            std::cerr << "Error: Cannot generate rotations for a shape with an empty row!" << std::endl;
            return;
        }
    }

    CelestialObject *first_rotation = this;
    CelestialObject *current_rotation = this;

    // Create right rotations
    for (int i = 0; i < 3; ++i) {
        vector<vector<bool>> rotated_shape = rotate_shape_right(current_rotation->shape);
        if (rotated_shape.empty()) {
            std::cerr << "Error: Failed to generate a valid rotation!" << std::endl;
            return;
        }

        auto *new_rotation = new CelestialObject(rotated_shape, object_type, starting_row, time_of_appearance);

        current_rotation->right_rotation = new_rotation;
        new_rotation->left_rotation = current_rotation;
        current_rotation = new_rotation;
    }

    // Complete the circular doubly linked list
    current_rotation->right_rotation = first_rotation;
    first_rotation->left_rotation = current_rotation;
}

