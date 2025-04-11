import pandas as pd
import random
from datetime import datetime, timedelta

def generate_financial_reports_data(num_companies=10, quarters=8):
    """Generate synthetic financial report data similar to what would be in Google Sheets"""
    companies = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
                    'JPM', 'BAC', 'WMT', 'XOM', 'BRK.B', 'JNJ', 'PG', 'V'
                ][:num_companies]

    # Base metrics for each company
    company_metrics = {}
    for company in companies:
        size_factor = random.uniform(0.5, 5)
        company_metrics[company] = {
            'revenue_base': size_factor * 10000,  # in millions
            'profit_margin': random.uniform(0.05, 0.35),
            'growth_rate': random.uniform(0.01, 0.15),
            'volatility': random.uniform(0.03, 0.12)
        }

    # Generate quarterly data
    quarters_list = []
    current_quarter = datetime.now().month // 3 + 1
    current_year = datetime.now().year

    for i in range(quarters):
        q = current_quarter - (i % 4)
        if q <= 0:
            q += 4
        y = current_year - (i // 4) - (1 if current_quarter - (i % 4) <= 0 else 0)
        quarters_list.append(f"Q{q} {y}")

    quarters_list.reverse()  # Oldest first

    # Generate data
    data = []
    for company in companies:
        metrics = company_metrics[company]
        revenue = metrics['revenue_base']

        for quarter in quarters_list:
            # Add some randomness and growth
            revenue_growth = 1 + metrics['growth_rate'] + random.uniform(-1, 1) * metrics['volatility']
            revenue *= revenue_growth

            # Calculate other metrics
            gross_profit = revenue * random.uniform(0.4, 0.8)
            operating_expenses = gross_profit * random.uniform(0.4, 0.8)
            operating_income = gross_profit - operating_expenses
            net_income = operating_income * metrics['profit_margin'] * random.uniform(0.8, 1.2)

            # Balance sheet items
            total_assets = revenue * random.uniform(1.5, 3)
            total_liabilities = total_assets * random.uniform(0.3, 0.7)
            total_equity = total_assets - total_liabilities

            # Cash flow items
            operating_cash_flow = net_income * random.uniform(0.8, 1.5)
            capex = revenue * random.uniform(0.05, 0.15)
            free_cash_flow = operating_cash_flow - capex

            data.append({
                'Company': company,
                'Quarter': quarter,
                'Revenue': round(revenue, 2),
                'Gross Profit': round(gross_profit, 2),
                'Operating Expenses': round(operating_expenses, 2),
                'Operating Income': round(operating_income, 2),
                'Net Income': round(net_income, 2),
                'Total Assets': round(total_assets, 2),
                'Total Liabilities': round(total_liabilities, 2),
                'Total Equity': round(total_equity, 2),
                'Operating Cash Flow': round(operating_cash_flow, 2),
                'Capital Expenditures': round(capex, 2),
                'Free Cash Flow': round(free_cash_flow, 2),
                'EPS': round(net_income / random.randint(1000, 10000), 2),
                'Dividend': round(net_income * random.uniform(0, 0.3) / random.randint(1000, 10000), 2)
            })

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    reports_df = generate_financial_reports_data()
    reports_df.to_excel('data/financial_reports.xlsx', index=False)
    print(f"Saved financial reports data: {len(reports_df)} records")