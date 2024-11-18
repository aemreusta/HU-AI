#include "AsteroidDash.h"
#include <fstream>
#include <sstream>

// Constructor to initialize AsteroidDash with the given parameters
AsteroidDash::AsteroidDash(const string &space_grid_file_name,
                           const string &celestial_objects_file_name,
                           const string &leaderboard_file_name,
                           const string &player_file_name,
                           const string &player_name)

        : leaderboard_file_name(leaderboard_file_name), leaderboard(Leaderboard()) {

    read_player(player_file_name, player_name);  // Initialize player using the player.dat file
    read_space_grid(space_grid_file_name);  // Initialize the grid after the player is loaded
    read_celestial_objects(celestial_objects_file_name);  // Load celestial objects
    leaderboard.read_from_file(leaderboard_file_name);
}

void AsteroidDash::read_space_grid(const string &input_file) {
    ifstream file(input_file);
    if (!file.is_open()) {
        cerr << "Error: Unable to open grid file: " << input_file << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        vector<int> row;
        stringstream ss(line);
        int cell;
        while (ss >> cell) {
            row.push_back(cell);
        }
        space_grid.push_back(row);
    }

    // Set grid dimensions
    grid_height = space_grid.size();
    grid_width = space_grid.empty() ? 0 : space_grid[0].size();

    file.close();
}

// Function to read the player from a file
void AsteroidDash::read_player(const string &player_file_name, const string &player_name) {
    ifstream file(player_file_name);
    if (!file.is_open()) {
        cerr << "Error: Unable to open player file: " << player_file_name << endl;
        return;
    }

    int row, col;
    file >> row >> col;

    vector<vector<bool>> shape;
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        vector<bool> row_vec;
        for (char ch : line) {
            if (ch == '1') row_vec.push_back(true);
            else if (ch == '0') row_vec.push_back(false);
        }
        shape.push_back(row_vec);
    }

    player = new Player(shape, row, col, player_name);

    file.close();
}

// Function to read celestial objects from a file
void AsteroidDash::read_celestial_objects(const string &input_file) {
    ifstream file(input_file);
    if (!file.is_open()) {
        cerr << "Error: Unable to open celestial objects file: " << input_file << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        // Parse celestial object data
        if (line.empty()) continue;

        vector<vector<bool>> shape;
        stringstream ss(line);
        char opening_char = ss.peek();

        if (opening_char == '[' || opening_char == '{') {
            vector<bool> row_vec;
            while (ss >> line && line[0] != 's') {
                row_vec.clear();
                for (char ch : line) {
                    if (ch == '1') row_vec.push_back(true);
                    else if (ch == '0') row_vec.push_back(false);
                }
                shape.push_back(row_vec);
            }
        }

        int starting_row, time_of_appearance;
        ObjectType type = (opening_char == '[') ? ASTEROID : (line.find("life") != string::npos ? LIFE_UP : AMMO);

        ss >> starting_row >> time_of_appearance;

        auto *new_object = new CelestialObject(shape, type, starting_row, time_of_appearance);
        new_object->next_celestial_object = celestial_objects_list_head;
        celestial_objects_list_head = new_object;
    }

    file.close();
}

// Print the entire space grid
void AsteroidDash::print_space_grid() const {
    cout << "Game Time: " << game_time << endl;
    cout << "Score: " << current_score << " Lives: " << player->lives << " Ammo: " << player->current_ammo << endl;

    for (const auto &row : space_grid) {
        for (int cell : row) {
            cout << (cell ? occupiedCellChar : unoccupiedCellChar);
        }
        cout << endl;
    }
}

// Function to update the space grid
void AsteroidDash::update_space_grid() {
    // Update celestial objects positions
    CelestialObject *current = celestial_objects_list_head;
    while (current) {
        // Update positions and check collisions
        current = current->next_celestial_object;
    }

    // Check for collisions with the player
    // TODO: Handle projectile interactions, collisions, and scoring logic
}

// Corresponds to the SHOOT command
void AsteroidDash::shoot() {
    if (player->current_ammo > 0) {
        player->current_ammo--;

        // Add projectile logic and update the grid
        // TODO: Implement projectile mechanics
    } else {
        cout << "No ammo to shoot!" << endl;
    }
}

// Destructor
AsteroidDash::~AsteroidDash() {
    delete player;

    while (celestial_objects_list_head) {
        CelestialObject *to_delete = celestial_objects_list_head;
        celestial_objects_list_head = celestial_objects_list_head->next_celestial_object;
        delete to_delete;
    }
}
