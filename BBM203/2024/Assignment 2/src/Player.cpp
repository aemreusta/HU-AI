#include "Player.h"

// Constructor to initialize the player's spacecraft, position, and ammo
Player::Player(const vector<vector<bool>> &shape, int row, int col, const string &player_name, int max_ammo, int lives)
        : spacecraft_shape(shape), position_row(row), position_col(col), player_name(player_name), max_ammo(max_ammo),
          current_ammo(max_ammo), lives(lives) {
    // Initialization complete
}

// Destructor to clean up any resources allocated for the player
Player::~Player() {
    // In this case, no dynamically allocated memory is used within the Player class.
    // If dynamic memory were used in the future, this destructor would ensure proper cleanup.
}

// Move player left within the grid boundaries
void Player::move_left() {
    if (position_col > 0) { // Ensure we don't go out of bounds
        position_col--;
    }
}

// Move player right within the grid boundaries
void Player::move_right(int grid_width) {
    if (position_col + spacecraft_shape[0].size() < grid_width) { // Ensure we don't exceed grid boundary
        position_col++;
    }
}

// Move player up within the grid boundaries
void Player::move_up() {
    if (position_row > 0) { // Ensure we don't go out of bounds
        position_row--;
    }
}

// Move player down within the grid boundaries
void Player::move_down(int grid_height) {
    if (position_row + spacecraft_shape.size() < grid_height) { // Ensure we don't exceed grid boundary
        position_row++;
    }
}
