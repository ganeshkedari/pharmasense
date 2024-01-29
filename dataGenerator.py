import os
import csv
from faker import Faker
import random
from datetime import datetime, timedelta
import us

fake = Faker()
# List of pharmaceutical brand names
pharmaceutical_brands = ["PharmaX", "MediCare", "HealthPlus", "BioWell", "MediGen", "VitaLife", "PharmaCare",
                         "WellnessRX", "LifeCure", "HealWell"]
customertype_sample = ["Academic", "Clinic", "Hospital", "Mid-Level Practitioner", "Nurse", "Pharmacist", "Practitioner"]
sales_type_sample = ["Online","Retail", "Non-Retail"]

def start_of_month(date):
    return date.replace(day=1)

# Generate 500 sales records
records = []
for _ in range(500):
    date = start_of_month(fake.date_between(start_date='-4y', end_date='today'))
    customer = fake.uuid4()
    customertype = random.choice(customertype_sample)
    sales_type = random.choice(sales_type_sample)
    product = random.choice(pharmaceutical_brands)
    country = "USA"
    state = random.choice(us.states.STATES).abbr
    city = fake.city()
    sales = random.randint(50, 500)*random.randint(50, 500)/random.randint(50, 500)

    records.append([date, customer, customertype, sales_type ,product, country, state, city, sales])

# Define the CSV file path
csv_file_path = './data/sampleSales.csv'

# Delete existing file if it exists
if os.path.exists(csv_file_path):
    os.remove(csv_file_path)
    print(f"Deleted existing file: {csv_file_path}")

# Write records to CSV file
header = ["Date", "Customer", "Customer Type","Sales Type", "Product", "Country", "State", "City", "Sales"]

with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(header)
    csv_writer.writerows(records)

print(f"CSV file '{csv_file_path}' with 5000 records has been generated.")
