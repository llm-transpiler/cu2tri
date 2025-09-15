# Pass@N Test Results Summary

## 📊 Model: gemini-2.5-pro

**Analysis Timestamp:** 20250912_192003  
**Total Cases:** 168 | **Max Attempts:** 10 | **Overall Success:** 146 (86.9%)

---

## 🎯 Pass@N Performance Indicators

| Metric | Value | Percentage |
|--------|-------|------------|
| **Pass@1** | 107/168 | 63.7% |
| **Pass@5** | 137/168 | 81.5% |
| **Pass@10** | 146/168 | 86.9% |
| **Overall Success** | 146/168 | 86.9% |
| **Avg Attempt Success Rate** | 0.6% | - |

---

## 📈 First Success Distribution

| Attempt | Cases | Percentage |
|---------|-------|------------|
| Attempt 1 | 107 | 63.7% |
| Attempt 2 | 14 | 8.3% |
| Attempt 3 | 9 | 5.4% |
| Attempt 4 | 6 | 3.6% |
| Attempt 5 | 1 | 0.6% |
| Attempt 6 | 4 | 2.4% |
| Attempt 7 | 2 | 1.2% |
| Attempt 8 | 2 | 1.2% |
| Attempt 9 | 1 | 0.6% |

---

## 📋 Case Type Performance (Pass@N)

| Status | Case Type | Pass@1 | Pass@5 | Pass@10 | Avg Success Rate | Performance |
|--------|-----------|--------|--------|---------|------------------|-------------|
| ❌ | `add` | 0.0% | 0.0% | 0.0% | 90.0% | Failed |
| ❌ | `avgpool` | 0.0% | 0.0% | 0.0% | 37.5% | Failed |
| ❌ | `bmm` | 0.0% | 0.0% | 0.0% | 47.5% | Failed |
| ❌ | `conv1d` | 0.0% | 0.0% | 0.0% | 71.2% | Failed |
| ❌ | `conv2d` | 0.0% | 0.0% | 0.0% | 36.2% | Failed |
| ❌ | `conv2dnchw` | 0.0% | 0.0% | 0.0% | 56.2% | Failed |
| ❌ | `deformable` | 0.0% | 0.0% | 0.0% | 13.8% | Failed |
| ❌ | `depthwiseconv` | 0.0% | 0.0% | 0.0% | 77.5% | Failed |
| ❌ | `gelu` | 0.0% | 0.0% | 0.0% | 0.0% | Failed |
| ❌ | `gemm` | 0.0% | 0.0% | 0.0% | 86.2% | Failed |
| ❌ | `gemv` | 0.0% | 0.0% | 0.0% | 92.5% | Failed |
| ❌ | `layernorm` | 0.0% | 0.0% | 0.0% | 23.8% | Failed |
| ❌ | `maxpool` | 0.0% | 0.0% | 0.0% | 33.8% | Failed |
| ❌ | `mha` | 0.0% | 0.0% | 0.0% | 5.0% | Failed |
| ❌ | `minpool` | 0.0% | 0.0% | 0.0% | 40.0% | Failed |
| ❌ | `relu` | 0.0% | 0.0% | 0.0% | 92.5% | Failed |
| ❌ | `rmsnorm` | 0.0% | 0.0% | 0.0% | 98.8% | Failed |
| ❌ | `sigmoid` | 0.0% | 0.0% | 0.0% | 80.0% | Failed |
| ❌ | `sign` | 0.0% | 0.0% | 0.0% | 88.8% | Failed |
| ❌ | `softmax` | 0.0% | 0.0% | 0.0% | 57.5% | Failed |
| ❌ | `sumpool` | 0.0% | 0.0% | 0.0% | 83.8% | Failed |

---

## 🔍 Detailed Analysis

### Top Performers (100% Success Rate)
- **add**: 8/8 cases
- **avgpool**: 8/8 cases
- **conv1d**: 8/8 cases
- **conv2d**: 8/8 cases
- **conv2dnchw**: 8/8 cases
- **depthwiseconv**: 8/8 cases
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
- **gelu**: 0/8 cases (0.0%)
- **mha**: 2/8 cases (25.0%)

### Pass@N Analysis Insights
- **107** cases (63.7%) succeeded on first attempt (Pass@1)
- **30** additional cases succeeded within 5 attempts (Pass@5)
- **22** cases (13.1%) failed after maximum attempts


---

## 📊 Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| **Excellent Case Types** (100%) | 17 | 81.0% |
| **Good Case Types** (80-99%) | 0 | 0.0% |
| **Moderate Case Types** (50-79%) | 2 | 9.5% |
| **Poor Case Types** (<50%) | 2 | 9.5% |

---

*Generated on 2025-09-15 at 09:28:33*
