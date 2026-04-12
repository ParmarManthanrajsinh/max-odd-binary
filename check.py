import benchmark
print(benchmark.bench_c('test', 'char *maximum_odd_binary(const char *s) { return strdup("10"); }'))
