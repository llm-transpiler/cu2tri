# Test Results Summary

## 📊 Model: openai/gpt-5

**Analysis Timestamp:** 20250911_042417  
**Total Cases:** 168 | **Success:** 162 (96.4%) | **Failed:** 6

---

## 🎯 Key Performance Indicators

| Metric | Value | Percentage |
|--------|-------|------------|
| **Round 1 Success** | 91/168 | 54.2% |
| **Round ≤5 Success** | 162/168 | 96.4% |
| **Overall Success** | 162/168 | 96.4% |
| **Case Types Fully Passed** | 15/21 | 71.4% |
| **Avg Rounds for Success** | 1.56 | - |

---

## 📈 Success Distribution by Round

| Round | Cases | Cumulative | Success Rate |
|-------|-------|------------|--------------|
| Round 1 | 91 | 91 | 54.2% |
| Round 2 | 55 | 146 | 86.9% |
| Round 3 | 13 | 159 | 94.6% |
| Round 4 | 2 | 161 | 95.8% |
| Round 5 | 1 | 162 | 96.4% |

---

## 📋 Case Type Performance

| Status | Case Type | Success Rate | Results | Performance |
|--------|-----------|--------------|---------|-------------|
| ✅ | `add` | 100.0% | 8/8 | Excellent |
| ✅ | `bmm` | 100.0% | 8/8 | Excellent |
| ✅ | `conv1d` | 100.0% | 8/8 | Excellent |
| ✅ | `conv2d` | 100.0% | 8/8 | Excellent |
| ✅ | `conv2dnchw` | 100.0% | 8/8 | Excellent |
| ✅ | `depthwiseconv` | 100.0% | 8/8 | Excellent |
| ✅ | `gemm` | 100.0% | 8/8 | Excellent |
| ✅ | `gemv` | 100.0% | 8/8 | Excellent |
| ✅ | `layernorm` | 100.0% | 8/8 | Excellent |
| ✅ | `minpool` | 100.0% | 8/8 | Excellent |
| ✅ | `relu` | 100.0% | 8/8 | Excellent |
| ✅ | `rmsnorm` | 100.0% | 8/8 | Excellent |
| ✅ | `sigmoid` | 100.0% | 8/8 | Excellent |
| ✅ | `sign` | 100.0% | 8/8 | Excellent |
| ✅ | `sumpool` | 100.0% | 8/8 | Excellent |
| 🟢 | `avgpool` | 87.5% | 7/8 | Good |
| 🟢 | `deformable` | 87.5% | 7/8 | Good |
| 🟢 | `gelu` | 87.5% | 7/8 | Good |
| 🟢 | `maxpool` | 87.5% | 7/8 | Good |
| 🟢 | `mha` | 87.5% | 7/8 | Good |
| 🟢 | `softmax` | 87.5% | 7/8 | Good |

---

## 🔍 Detailed Analysis

### Top Performers (100% Success Rate)
- **add**: 8/8 cases
- **bmm**: 8/8 cases
- **conv1d**: 8/8 cases
- **conv2d**: 8/8 cases
- **conv2dnchw**: 8/8 cases
- **depthwiseconv**: 8/8 cases
- **gemm**: 8/8 cases
- **gemv**: 8/8 cases
- **layernorm**: 8/8 cases
- **minpool**: 8/8 cases
- **relu**: 8/8 cases
- **rmsnorm**: 8/8 cases
- **sigmoid**: 8/8 cases
- **sign**: 8/8 cases
- **sumpool**: 8/8 cases

### Areas for Improvement
- *All case types achieved ≥50% success rate*

### Round Analysis Insights
- **91** cases (54.2%) succeeded on first attempt
- **71** additional cases succeeded within 5 rounds
- **6** cases (3.6%) failed after maximum rounds


---

## 📊 Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| **Excellent Case Types** (100%) | 15 | 71.4% |
| **Good Case Types** (80-99%) | 6 | 28.6% |
| **Moderate Case Types** (50-79%) | 0 | 0.0% |
| **Poor Case Types** (<50%) | 0 | 0.0% |

---

*Generated on 2025-09-15 at 09:26:54*
