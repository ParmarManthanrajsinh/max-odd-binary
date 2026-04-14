std::string maximum_odd_binary(std::string s) {
  int len = s.size();
  int ones = 0;

  for (char c : s)
    ones += (c == '1');

  if (ones == 0)
    return "";
  std::string result(len, '0');
  std::fill(result.begin(), result.begin() + (ones - 1), '1');
  result[len - 1] = '1';
  return result;
}
