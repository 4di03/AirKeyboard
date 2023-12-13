#include <iostream>

int main() {
    // Intentional memory error
    int *arr = new int[10];
    arr[10] = 42;  // Writing beyond the allocated memory

    // Deallocate memory to avoid leaks
    delete[] arr;

    return 0;
}