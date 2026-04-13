import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from src.analyze_user import analyze_user

print("=== TEST 1: Normal high-risk user ===")
income  = [30000 + np.random.normal(0, 1500) for _ in range(24)]
expense = [28000 + np.random.normal(0, 1000) for _ in range(24)]
result  = analyze_user(income, expense, investment_amount=50000)
assert isinstance(result['risk_label'], str)
assert result['risk_label'] in ['High', 'Medium', 'Low', 'Unknown']
assert result['expense_trend'] in ['Improving', 'Stable', 'Deteriorating', 'Forecast Error', 'Insufficient Data']
assert isinstance(result['mc_summary'], str) and len(result['mc_summary']) > 10
print(f"  Risk: {result['risk_label']} | Expense: {result['expense_trend']} | Savings: {result['savings_trend']}")
print(f"  MC: {result['mc_summary']}")

print("\n=== TEST 2: Flat — expense equals income ===")
result2 = analyze_user([30000.0]*24, [30000.0]*24, investment_amount=10000)
assert result2['risk_label'] in ['High', 'Medium', 'Low', 'Unknown']
print(f"  Risk: {result2['risk_label']} | Expense trend: {result2['expense_trend']}")

print("\n=== TEST 3: Expense greater than income ===")
result3 = analyze_user([30000.0]*24, [35000.0]*24, investment_amount=10000)
assert result3['risk_label'] in ['High', 'Medium', 'Low', 'Unknown']
print(f"  Risk: {result3['risk_label']} | Savings trend: {result3['savings_trend']}")

print("\n=== TEST 4: Validation errors fire correctly ===")
try:
    analyze_user([50000]*24, [30000]*10, investment_amount=10000)
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"  Caught expected error: {e}")

try:
    analyze_user([50000]*24, [30000]*24, investment_amount=-1000)
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"  Caught expected error: {e}")

print("\n=== ALL TESTS PASSED ===")
