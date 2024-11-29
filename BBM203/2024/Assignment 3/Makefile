# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall

# Directories
SRC_DIR = src
IO_DIR = io
BUILD_DIR = build

# Source files and object files
SRC_FILES = $(SRC_DIR)/Map.cpp $(SRC_DIR)/Stack.cpp $(SRC_DIR)/Queue.cpp $(SRC_DIR)/RoutePlanner.cpp $(SRC_DIR)/main.cpp
OBJ_FILES = $(SRC_FILES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Output zip file name
ZIP_NAME = b2200765036.zip

# Output
TARGET = $(BUILD_DIR)/DroneALPHA

# Rule to create the build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule to compile .cpp files into .o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(info Compiling $< to $@)  # Debug message
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Linking the object files into the final executable
$(TARGET): $(OBJ_FILES)
	$(info Linking $(OBJ_FILES) into $(TARGET))  # Debug message
	$(CXX) $(CXXFLAGS) $^ -o $@

# Clean up (only removes build-related files)
clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)

# Rule to zip all .cpp and .h files except main.cpp, and avoid including src in the zip
zip:
	cd $(SRC_DIR) && zip -r ../$(ZIP_NAME) ./*.cpp ./*.h -x ./main.cpp

# Run the program with command-line arguments
run: $(TARGET)
	$(info Running the program)  # Debug message
	./$(TARGET) 250 5 $(IO_DIR)/distance_data.csv $(IO_DIR)/priority_provinces.txt $(IO_DIR)/weather_restricted_provinces.txt

.PHONY: clean run
