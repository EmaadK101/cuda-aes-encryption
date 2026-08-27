#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>
#include <windows.h>

// Multiplication by 2 in GF(2^8)
__device__ unsigned char mul2(unsigned char a) {
    return (a & 0x80) ? ((a << 1) ^ 0x1b) : (a << 1);
}

// Multiplication by 3 in GF(2^8)
__device__ unsigned char mul3(unsigned char a) {
    return mul2(a) ^ a;
}

// AES S-box
__constant__ unsigned char d_sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

// AES round constants
__constant__ unsigned char d_rcon[10] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

// T-tables for optimized AES implementation
__constant__ unsigned int d_T0[256], d_T1[256], d_T2[256], d_T3[256];

// Host version of mul2 function for T-table initialization
unsigned char mul2_host(unsigned char a) {
    return (a & 0x80) ? ((a << 1) ^ 0x1b) : (a << 1);
}

// Initialize T-tables
void initTTables() {
    unsigned int T0[256], T1[256], T2[256], T3[256];
    unsigned char sbox_host[256];
    
    // First copy the S-box to host memory
    cudaMemcpyFromSymbol(sbox_host, d_sbox, sizeof(sbox_host));
   
    // Generate T-tables using the host copy
    for (int i = 0; i < 256; i++) {
        unsigned char s = sbox_host[i];
        T0[i] = (s << 24) | (s << 16) | (s << 8) | mul2_host(s);
        T1[i] = (s << 16) | (s << 8) | (s) | (mul2_host(s) << 24);
        T2[i] = (s << 8) | (s) | (mul2_host(s) << 24) | (s << 16);
        T3[i] = (s) | (mul2_host(s) << 24) | (s << 16) | (s << 8);
    }
   
    // Copy T-tables to constant memory
    cudaMemcpyToSymbol(d_T0, T0, sizeof(T0));
    cudaMemcpyToSymbol(d_T1, T1, sizeof(T1));
    cudaMemcpyToSymbol(d_T2, T2, sizeof(T2));
    cudaMemcpyToSymbol(d_T3, T3, sizeof(T3));
}

// Key expansion for 32-bit word based implementation
__device__ void expandKey(unsigned int* expandedKey, const unsigned char* key) {
    // First round key is the key itself
    for (int i = 0; i < 4; i++) {
        expandedKey[i] = ((unsigned int)key[4*i] << 24) |
                         ((unsigned int)key[4*i+1] << 16) |
                         ((unsigned int)key[4*i+2] << 8) |
                         ((unsigned int)key[4*i+3]);
    }
   
    // Generate the remaining round keys
    for (int i = 4; i < 44; i++) {
        unsigned int temp = expandedKey[i-1];
       
        if (i % 4 == 0) {
            // RotWord
            temp = (temp << 8) | (temp >> 24);
           
            // SubWord
            temp = ((unsigned int)d_sbox[(temp >> 24) & 0xFF] << 24) |
                   ((unsigned int)d_sbox[(temp >> 16) & 0xFF] << 16) |
                   ((unsigned int)d_sbox[(temp >> 8) & 0xFF] << 8) |
                   ((unsigned int)d_sbox[temp & 0xFF]);
           
            // XOR with Rcon
            temp ^= ((unsigned int)d_rcon[i/4-1] << 24);
        }
       
        expandedKey[i] = expandedKey[i-4] ^ temp;
    }
}

// Optimized AES encryption kernel using T-tables
__global__ void aes_ctr_encrypt_kernel_optimized(unsigned char* plaintext, unsigned char* ciphertext,
                                               const unsigned char* key, const unsigned int* counter,
                                               int numBlocks, int bytesPerThread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;
   
    // Shared memory for expanded key
    __shared__ unsigned int expandedKey[44];  // 11 round keys (4 words each)
   
    // Only the first thread in a block expands the key
    if (threadIdx.x == 0) {
        // Key expansion code here (optimized for 32-bit words)
        expandKey(expandedKey, key);
    }
   
    __syncthreads();
   
    // Process multiple blocks per thread for better performance
    for (int i = tid; i < numBlocks; i += totalThreads) {
        // Create counter value for this block
        unsigned int ctr[4];
        ctr[0] = counter[0];
        ctr[1] = counter[1];
        ctr[2] = counter[2];
        ctr[3] = counter[3] + i;  // Increment counter
       
        // T-table based AES encryption of counter
        unsigned int s0, s1, s2, s3;
        unsigned int t0, t1, t2, t3;
       
        // Initial round key addition
        s0 = ctr[0] ^ expandedKey[0];
        s1 = ctr[1] ^ expandedKey[1];
        s2 = ctr[2] ^ expandedKey[2];
        s3 = ctr[3] ^ expandedKey[3];
       
        // Main rounds (unrolled for performance)
        #pragma unroll
        for (int round = 1; round < 10; round++) {
            t0 = d_T0[(s0 >> 24) & 0xFF] ^ d_T1[(s1 >> 16) & 0xFF] ^
                 d_T2[(s2 >> 8) & 0xFF] ^ d_T3[s3 & 0xFF] ^ expandedKey[round*4];
            t1 = d_T0[(s1 >> 24) & 0xFF] ^ d_T1[(s2 >> 16) & 0xFF] ^
                 d_T2[(s3 >> 8) & 0xFF] ^ d_T3[s0 & 0xFF] ^ expandedKey[round*4+1];
            t2 = d_T0[(s2 >> 24) & 0xFF] ^ d_T1[(s3 >> 16) & 0xFF] ^
                 d_T2[(s0 >> 8) & 0xFF] ^ d_T3[s1 & 0xFF] ^ expandedKey[round*4+2];
            t3 = d_T0[(s3 >> 24) & 0xFF] ^ d_T1[(s0 >> 16) & 0xFF] ^
                 d_T2[(s1 >> 8) & 0xFF] ^ d_T3[s2 & 0xFF] ^ expandedKey[round*4+3];
            s0 = t0;
            s1 = t1;
            s2 = t2;
            s3 = t3;
        }
       
        // Final round (uses S-box instead of T-tables)
        unsigned char keystream[16];
       
        // Final round substitution and shift rows (combined)
        keystream[0] = d_sbox[(s0 >> 24) & 0xFF];
        keystream[1] = d_sbox[(s1 >> 16) & 0xFF];
        keystream[2] = d_sbox[(s2 >> 8) & 0xFF];
        keystream[3] = d_sbox[s3 & 0xFF];
       
        keystream[4] = d_sbox[(s1 >> 24) & 0xFF];
        keystream[5] = d_sbox[(s2 >> 16) & 0xFF];
        keystream[6] = d_sbox[(s3 >> 8) & 0xFF];
        keystream[7] = d_sbox[s0 & 0xFF];
       
        keystream[8] = d_sbox[(s2 >> 24) & 0xFF];
        keystream[9] = d_sbox[(s3 >> 16) & 0xFF];
        keystream[10] = d_sbox[(s0 >> 8) & 0xFF];
        keystream[11] = d_sbox[s1 & 0xFF];
       
        keystream[12] = d_sbox[(s3 >> 24) & 0xFF];
        keystream[13] = d_sbox[(s0 >> 16) & 0xFF];
        keystream[14] = d_sbox[(s1 >> 8) & 0xFF];
        keystream[15] = d_sbox[s2 & 0xFF];
       
        // Add final round key
        for (int j = 0; j < 4; j++) {
            ((unsigned int*)keystream)[j] ^= expandedKey[40+j];
        }
       
        // XOR keystream with plaintext to produce ciphertext
        int offset = i * 16;
        for (int j = 0; j < 16 && offset + j < numBlocks * 16; j++) {
            ciphertext[offset + j] = plaintext[offset + j] ^ keystream[j];
        }
    }
}

// Function to get current time in seconds with microsecond precision
double get_time() {
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    return start.QuadPart / (double)frequency.QuadPart;
}

// Host function to perform AES-CTR encryption
void aes_ctr_encrypt_optimized(const unsigned char* plaintext, unsigned char* ciphertext,
                             const unsigned char* key, const unsigned char* iv, size_t size) {
    unsigned char *d_plaintext, *d_ciphertext, *d_key;
    unsigned int *d_counter;
   
    // Calculate number of blocks
    int numBlocks = (size + 15) / 16;
   
    // Initialize T-tables
    initTTables();
   
    // Prepare counter from IV
    unsigned int counter[4];
    for (int i = 0; i < 4; i++) {
        counter[i] = ((unsigned int)iv[4*i] << 24) |
                     ((unsigned int)iv[4*i+1] << 16) |
                     ((unsigned int)iv[4*i+2] << 8) |
                     ((unsigned int)iv[4*i+3]);
    }
   
    // Allocate device memory
    cudaMalloc((void**)&d_plaintext, size);
    cudaMalloc((void**)&d_ciphertext, size);
    cudaMalloc((void**)&d_key, 16);
    cudaMalloc((void**)&d_counter, 16);
   
    // Copy data to device
    cudaMemcpy(d_plaintext, plaintext, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counter, counter, 16, cudaMemcpyHostToDevice);
   
    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;
   
    // Determine bytes per thread for load balancing
    int bytesPerThread = 16;  // Default to one block per thread
   
    // Launch kernel
    aes_ctr_encrypt_kernel_optimized<<<blocksPerGrid, threadsPerBlock>>>(
        d_plaintext, d_ciphertext, d_key, d_counter, numBlocks, bytesPerThread);
   
    // Copy result back to host
    cudaMemcpy(ciphertext, d_ciphertext, size, cudaMemcpyDeviceToHost);
   
    // Free device memory
    cudaFree(d_plaintext);
    cudaFree(d_ciphertext);
    cudaFree(d_key);
    cudaFree(d_counter);
}

// Add this to your existing code
int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s [file to encrypt] [key file] [number of iterations]\n", argv[0]);
        return 1;
    }
    
    // Parse number of iterations
    int iterations = atoi(argv[3]);
    if (iterations <= 0) {
        printf("Number of iterations must be positive\n");
        return 1;
    }
    
    // Read input file
    FILE* inputFile = fopen(argv[1], "rb");
    if (!inputFile) {
        printf("Error opening input file\n");
        return 1;
    }
    
    // Get file size
    fseek(inputFile, 0, SEEK_END);
    size_t fileSize = ftell(inputFile);
    fseek(inputFile, 0, SEEK_SET);
    
    // Allocate memory for plaintext
    unsigned char* plaintext = (unsigned char*)malloc(fileSize);
    if (!plaintext) {
        printf("Memory allocation failed\n");
        fclose(inputFile);
        return 1;
    }
    
    // Read input file
    fread(plaintext, 1, fileSize, inputFile);
    fclose(inputFile);
    
    // Read key file
    FILE* keyFile = fopen(argv[2], "rb");
    if (!keyFile) {
        printf("Error opening key file\n");
        free(plaintext);
        return 1;
    }
    
    // Read key (AES-128 uses 16-byte key)
    unsigned char key[16];
    if (fread(key, 1, 16, keyFile) != 16) {
        printf("Error reading key file or key size incorrect\n");
        fclose(keyFile);
        free(plaintext);
        return 1;
    }
    fclose(keyFile);
    
    // Generate IV (for simplicity, using a fixed IV here)
    unsigned char iv[16] = {0}; // In a real application, use a secure random IV
    
    // Allocate memory for ciphertext and decrypted text
    unsigned char* ciphertext = (unsigned char*)malloc(fileSize);
    unsigned char* decrypted = (unsigned char*)malloc(fileSize);
    if (!ciphertext || !decrypted) {
        printf("Memory allocation failed\n");
        free(plaintext);
        if (ciphertext) free(ciphertext);
        if (decrypted) free(decrypted);
        return 1;
    }
    
    // Get base filename without extension
    char baseFilename[256] = {0};
    strcpy(baseFilename, argv[1]);
    char* extension = strrchr(baseFilename, '.');
    if (extension) *extension = '\0';
    
    // Create CSV files for timing data
    FILE* encryptionTimeFile = fopen("encryption_time.csv", "w");
    FILE* decryptionTimeFile = fopen("decryption_time.csv", "w");
    
    if (!encryptionTimeFile || !decryptionTimeFile) {
        printf("Error creating CSV files\n");
        if (encryptionTimeFile) fclose(encryptionTimeFile);
        if (decryptionTimeFile) fclose(decryptionTimeFile);
        free(plaintext);
        free(ciphertext);
        free(decrypted);
        return 1;
    }
    
    // Write CSV headers
    fprintf(encryptionTimeFile, "iteration,time\n");
    fprintf(decryptionTimeFile, "iteration,time\n");
    
    // Run the encryption and decryption multiple times
    for (int i = 1; i <= iterations; i++) {
        // Create filenames for this iteration
        char encryptedFilename[300], decryptedFilename[300];
        sprintf(encryptedFilename, "%s_e%d.txt", baseFilename, i);
        sprintf(decryptedFilename, "%s_d%d.txt", baseFilename, i);
        
        // Time encryption
        double start_encrypt = get_time();
        aes_ctr_encrypt_optimized(plaintext, ciphertext, key, iv, fileSize);
        double end_encrypt = get_time();
        double encrypt_time = end_encrypt - start_encrypt;
        
        // Write encrypted data to output file
        FILE* encryptedFile = fopen(encryptedFilename, "wb");
        if (!encryptedFile) {
            printf("Error creating encrypted output file\n");
            fclose(encryptionTimeFile);
            fclose(decryptionTimeFile);
            free(plaintext);
            free(ciphertext);
            free(decrypted);
            return 1;
        }
        fwrite(ciphertext, 1, fileSize, encryptedFile);
        fclose(encryptedFile);
        
        // Time decryption
        double start_decrypt = get_time();
        aes_ctr_encrypt_optimized(ciphertext, decrypted, key, iv, fileSize);
        double end_decrypt = get_time();
        double decrypt_time = end_decrypt - start_decrypt;
        
        // Write decrypted data to output file
        FILE* decryptedFile = fopen(decryptedFilename, "wb");
        if (!decryptedFile) {
            printf("Error creating decrypted output file\n");
            fclose(encryptionTimeFile);
            fclose(decryptionTimeFile);
            free(plaintext);
            free(ciphertext);
            free(decrypted);
            return 1;
        }
        fwrite(decrypted, 1, fileSize, decryptedFile);
        fclose(decryptedFile);
        
        // Write timing data to CSV files
        fprintf(encryptionTimeFile, "%d,%.6f\n", i, encrypt_time);
        fprintf(decryptionTimeFile, "%d,%.6f\n", i, decrypt_time);
        
        // Print timing information to console
        printf("Iteration %d:\n", i);
        printf("  Encryption time: %.6f seconds\n", encrypt_time);
        printf("  Decryption time: %.6f seconds\n", decrypt_time);
        printf("  Files saved as %s and %s\n", encryptedFilename, decryptedFilename);
    }
    
    // Close CSV files
    fclose(encryptionTimeFile);
    fclose(decryptionTimeFile);
    printf("Timing data saved to encryption_time.csv and decryption_time.csv\n");
    
    // Clean up
    free(plaintext);
    free(ciphertext);
    free(decrypted);
    
    return 0;
}
