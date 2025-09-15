# Test Results Summary

## 📊 Model: deepseek-reasoner

**Analysis Timestamp:** 20250910_210211  
**Total Cases:** 168 | **Success:** 156 (92.9%) | **Failed:** 12

---

## 🎯 Key Performance Indicators

| Metric | Value | Percentage |
|--------|-------|------------|
| **Round 1 Success** | 64/168 | 38.1% |
| **Round ≤5 Success** | 156/168 | 92.9% |
| **Overall Success** | 156/168 | 92.9% |
| **Case Types Fully Passed** | 17/21 | 81.0% |
| **Avg Rounds for Success** | 1.89 | - |

---

## 📈 Success Distribution by Round

| Round | Cases | Cumulative | Success Rate |
|-------|-------|------------|--------------|
| Round 1 | 64 | 64 | 38.1% |
| Round 2 | 60 | 124 | 73.8% |
| Round 3 | 21 | 145 | 86.3% |
| Round 4 | 7 | 152 | 90.5% |
| Round 5 | 4 | 156 | 92.9% |

---

## 📋 Case Type Performance

| Status | Case Type | Success Rate | Results | Performance |
|--------|-----------|--------------|---------|-------------|
| ✅ | `add` | 100.0% | 8/8 | Excellent |
| ✅ | `avgpool` | 100.0% | 8/8 | Excellent |
| ✅ | `conv1d` | 100.0% | 8/8 | Excellent |
| ✅ | `conv2d` | 100.0% | 8/8 | Excellent |
| ✅ | `conv2dnchw` | 100.0% | 8/8 | Excellent |
| ✅ | `gelu` | 100.0% | 8/8 | Excellent |
| ✅ | `gemm` | 100.0% | 8/8 | Excellent |
| ✅ | `gemv` | 100.0% | 8/8 | Excellent |
| ✅ | `layernorm` | 100.0% | 8/8 | Excellent |
| ✅ | `maxpool` | 100.0% | 8/8 | Excellent |
| ✅ | `minpool` | 100.0% | 8/8 | Excellent |
| ✅ | `relu` | 100.0% | 8/8 | Excellent |
| ✅ | `rmsnorm` | 100.0% | 8/8 | Excellent |
| ✅ | `sigmoid` | 100.0% | 8/8 | Excellent |
| ✅ | `sign` | 100.0% | 8/8 | Excellent |
| ✅ | `softmax` | 100.0% | 8/8 | Excellent |
| ✅ | `sumpool` | 100.0% | 8/8 | Excellent |
| 🟡 | `depthwiseconv` | 75.0% | 6/8 | Moderate |
| 🟡 | `bmm` | 62.5% | 5/8 | Moderate |
| 🟡 | `mha` | 62.5% | 5/8 | Moderate |
| 🟡 | `deformable` | 50.0% | 4/8 | Moderate |

---

## 🔍 Detailed Analysis

### Top Performers (100% Success Rate)
- **add**: 8/8 cases
- **avgpool**: 8/8 cases
- **conv1d**: 8/8 cases
- **conv2d**: 8/8 cases
- **conv2dnchw**: 8/8 cases
- **gelu**: 8/8 cases
- **gemm**: 8/8 cases
- **gemv**: 8/8 cases
- **layernorm**: 8/8 cases
- **maxpool**: 8/8 cases
- **minpool**: 8/8 cases
- **relu**: 8/8 cases
- **rmsnorm**: 8/8 cases
- **sigmoid**: 8/8 cases
- **sign**: 8/8 cases
- **softmax**: 8/8 cases
- **sumpool**: 8/8 cases

### Areas for Improvement
- *All case types achieved ≥50% success rate*

### Round Analysis Insights
- **64** cases (38.1%) succeeded on first attempt
- **92** additional cases succeeded within 5 rounds
- **12** cases (7.1%) failed after maximum rounds


---

## 📊 Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| **Excellent Case Types** (100%) | 17 | 81.0% |
| **Good Case Types** (80-99%) | 0 | 0.0% |
| **Moderate Case Types** (50-79%) | 4 | 19.0% |
| **Poor Case Types** (<50%) | 0 | 0.0% |

---

*Generated on 2025-09-15 at 09:27:21*
