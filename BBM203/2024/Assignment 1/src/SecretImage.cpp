#include "SecretImage.h"


// Constructor: split image into upper and lower triangular arrays
SecretImage::SecretImage(const GrayscaleImage& image) 
    : width(image.get_width()), height(image.get_height()) {

    // Upper ve lower üçgen matrisleri için bellek tahsisi
    upper_triangular = new int[width * height / 2];
    lower_triangular = new int[width * height / 2];

    // Piksel verilerini üst ve alt üçgen matrislere böl
    int upper_index = 0;
    int lower_index = 0;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (j >= i) {  // Üst üçgende olanlar
                upper_triangular[upper_index++] = image.get_pixel(i, j);
            } else {  // Alt üçgende olanlar
                lower_triangular[lower_index++] = image.get_pixel(i, j);
            }
        }
    }
}

// Constructor: instantiate based on data read from file
SecretImage::SecretImage(int w, int h, int * upper, int * lower) 
    : width(w), height(h) {

    // Bellek tahsisi ve kopyalama işlemi
    int upper_size = w * h / 2;
    int lower_size = w * h / 2;
    
    upper_triangular = new int[upper_size];
    lower_triangular = new int[lower_size];

    std::copy(upper, upper + upper_size, upper_triangular);
    std::copy(lower, lower + lower_size, lower_triangular);
}

// Destructor: free the arrays
SecretImage::~SecretImage() {
    delete[] upper_triangular;
    delete[] lower_triangular;
}

// Reconstructs and returns the full image from upper and lower triangular matrices.
GrayscaleImage SecretImage::reconstruct() const {
    GrayscaleImage image(width, height);

    int upper_index = 0;
    int lower_index = 0;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (j >= i) {
                image.set_pixel(i, j, upper_triangular[upper_index++]);
            } else {
                image.set_pixel(i, j, lower_triangular[lower_index++]);
            }
        }
    }
    return image;
}

// Save the filtered image back to the triangular arrays
void SecretImage::save_back(const GrayscaleImage& image) {
    int upper_index = 0;
    int lower_index = 0;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (j >= i) {
                upper_triangular[upper_index++] = image.get_pixel(i, j);
            } else {
                lower_triangular[lower_index++] = image.get_pixel(i, j);
            }
        }
    }
}

// Save the upper and lower triangular arrays to a file
void SecretImage::save_to_file(const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("File could not be opened.");
    }

    file << width << " " << height << "\n";

    for (int i = 0; i < width * height / 2; ++i) {
        file << upper_triangular[i] << " ";
    }
    file << "\n";

    for (int i = 0; i < width * height / 2; ++i) {
        file << lower_triangular[i] << " ";
    }
    file << "\n";
}

// Static function to load a SecretImage from a file
SecretImage SecretImage::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("File could not be opened.");
    }

    int w, h;
    file >> w >> h;

    int upper_size = w * h / 2;
    int lower_size = w * h / 2;

    int *upper = new int[upper_size];
    int *lower = new int[lower_size];

    for (int i = 0; i < upper_size; ++i) {
        file >> upper[i];
    }

    for (int i = 0; i < lower_size; ++i) {
        file >> lower[i];
    }

    file.close();

    SecretImage secret_image(nullptr);
    return secret_image;
}

// Returns a pointer to the upper triangular part of the secret image.
int * SecretImage::get_upper_triangular() const {
    return upper_triangular;
}

// Returns a pointer to the lower triangular part of the secret image.
int * SecretImage::get_lower_triangular() const {
    return lower_triangular;
}

// Returns the width of the secret image.
int SecretImage::get_width() const {
    return width;
}

// Returns the height of the secret image.
int SecretImage::get_height() const {
    return height;
}
