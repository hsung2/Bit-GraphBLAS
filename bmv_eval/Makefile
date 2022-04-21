CC = nvcc

FLAGS = -arch=sm_60 -O3 -std=c++11 -w
LINK = -lcublas -lcusparse -lcudart

all: baseline bmv_bin_bin_full bmv_bin_full_full bmv_bin_bin_bin

# baseline
baseline: backend/bsrbmv.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) test_bin_bin_full.cu -DTILEDIM=4 -DBASELINE -o bmv/$@

# bmv_bin_bin_full
bmv_bin_bin_full: bmv4_bin_bin_full bmv8_bin_bin_full bmv16_bin_bin_full bmv32_bin_bin_full
bmv4_bin_bin_full: backend/bsrbmv.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) test_bin_bin_full.cu -DTILEDIM=4 -o bmv/$@
bmv8_bin_bin_full: backend/bsrbmv.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) test_bin_bin_full.cu -DTILEDIM=8 -o bmv/$@
bmv16_bin_bin_full: backend/bsrbmv.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) test_bin_bin_full.cu -DTILEDIM=16 -o bmv/$@
bmv32_bin_bin_full: backend/bsrbmv.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) test_bin_bin_full.cu -DTILEDIM=32 -o bmv/$@


# bmv_bin_full_full
bmv_bin_full_full: bmv4_bin_full_full bmv8_bin_full_full bmv16_bin_full_full bmv32_bin_full_full
bmv4_bin_full_full: backend/bsrbmv.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) test_bin_full_full.cu -DTILEDIM=4 -o bmv/$@
bmv8_bin_full_full: backend/bsrbmv.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) test_bin_full_full.cu -DTILEDIM=8 -o bmv/$@
bmv16_bin_full_full: backend/bsrbmv.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) test_bin_full_full.cu -DTILEDIM=16 -o bmv/$@
bmv32_bin_full_full: backend/bsrbmv.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) test_bin_full_full.cu -DTILEDIM=32 -o bmv/$@


# bmv_bin_bin_bin
bmv_bin_bin_bin: bmv4_bin_bin_bin bmv8_bin_bin_bin bmv16_bin_bin_bin bmv32_bin_bin_bin
bmv4_bin_bin_bin: backend/bsrbmv.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) test_bin_bin_bin.cu -DTILEDIM=4 -o bmv/$@
bmv8_bin_bin_bin: backend/bsrbmv.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) test_bin_bin_bin.cu -DTILEDIM=8 -o bmv/$@
bmv16_bin_bin_bin: backend/bsrbmv.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) test_bin_bin_bin.cu -DTILEDIM=16 -o bmv/$@
bmv32_bin_bin_bin: backend/bsrbmv.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) test_bin_bin_bin.cu -DTILEDIM=32 -o bmv/$@


clean:
	rm -f bmv/*