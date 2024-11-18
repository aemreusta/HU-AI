#include "GameController.h"
#include <fstream>
#include <iostream>

// Simply instantiates the game
GameController::GameController(
        const string &space_grid_file_name,
        const string &celestial_objects_file_name,
        const string &leaderboard_file_name,
        const string &player_file_name,
        const string &player_name
) {
    game = new AsteroidDash(space_grid_file_name, celestial_objects_file_name, leaderboard_file_name, player_file_name,
                            player_name);
}

// Reads commands from the given input file, executes each command in a game tick
void GameController::play(const string &commands_file) {
    ifstream file(commands_file);

    if (!file.is_open()) {
        cerr << "Error: Unable to open commands file: " << commands_file << endl;
        return;
    }

    string command;
    while (getline(file, command)) {
        // Execute each command dynamically in real-time
        if (command == "MOVE_UP") {
            game->player->move_up();
        } else if (command == "MOVE_DOWN") {
            game->player->move_down(game->grid_height);
        } else if (command == "MOVE_LEFT") {
            game->player->move_left();
        } else if (command == "MOVE_RIGHT") {
            game->player->move_right(game->grid_width);
        } else if (command == "SHOOT") {
            game->shoot();
        } else if (command == "PRINT_GRID") {
            game->print_space_grid();
        } else if (command == "NOP") {
            // No operation; continue to next tick
        } else {
            cerr << "Unknown command: " << command << endl;
        }

        // Update game state for the next tick
        game->update_space_grid();

        // Check for game over conditions
        if (game->game_over) {
            break;
        }
    }

    file.close();
}

// Destructor to delete dynamically allocated member variables
GameController::~GameController() {
    delete game; // Free the dynamically allocated AsteroidDash instance
}
