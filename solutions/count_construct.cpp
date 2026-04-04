auto maximum_odd_binary(std::string s) -> std::string {
  auto n = std::ranges::count(s, '1');
  return std::string(n - 1, '1') + std::string(s.size() - n, '0') + '1';
}
