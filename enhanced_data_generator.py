"""
Enhanced India-Wide Cybercrime Data Generator
Covers all states, major cities, and realistic cybercrime patterns
"""

import random
import json
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

# Comprehensive Indian cities and states data
INDIAN_STATES_CITIES = {
    "Andhra Pradesh": {
        "major_cities": ["Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Kurnool", "Rajahmundry", "Tirupati", "Kadapa"],
        "coordinates": {"lat_range": (12.5, 19.5), "lng_range": (77.0, 84.7)},
        "crime_rate_factor": 0.7
    },
    "Arunachal Pradesh": {
        "major_cities": ["Itanagar", "Naharlagun", "Pasighat", "Tezpur", "Bomdila"],
        "coordinates": {"lat_range": (26.3, 29.4), "lng_range": (91.7, 97.4)},
        "crime_rate_factor": 0.2
    },
    "Assam": {
        "major_cities": ["Guwahati", "Silchar", "Dibrugarh", "Jorhat", "Nagaon", "Tinsukia", "Tezpur"],
        "coordinates": {"lat_range": (24.1, 27.9), "lng_range": (89.7, 96.0)},
        "crime_rate_factor": 0.5
    },
    "Bihar": {
        "major_cities": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Purnia", "Darbhanga", "Bihar Sharif", "Arrah"],
        "coordinates": {"lat_range": (24.2, 27.5), "lng_range": (83.3, 88.1)},
        "crime_rate_factor": 0.6
    },
    "Chhattisgarh": {
        "major_cities": ["Raipur", "Bhilai", "Korba", "Bilaspur", "Durg", "Rajnandgaon"],
        "coordinates": {"lat_range": (17.8, 24.1), "lng_range": (80.2, 84.4)},
        "crime_rate_factor": 0.4
    },
    "Goa": {
        "major_cities": ["Panaji", "Margao", "Vasco da Gama", "Mapusa", "Ponda"],
        "coordinates": {"lat_range": (15.0, 15.8), "lng_range": (73.7, 74.3)},
        "crime_rate_factor": 0.3
    },
    "Gujarat": {
        "major_cities": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Jamnagar", "Junagadh", "Gandhinagar"],
        "coordinates": {"lat_range": (20.1, 24.7), "lng_range": (68.2, 74.5)},
        "crime_rate_factor": 0.8
    },
    "Haryana": {
        "major_cities": ["Faridabad", "Gurgaon", "Panipat", "Ambala", "Yamunanagar", "Rohtak", "Hisar", "Karnal"],
        "coordinates": {"lat_range": (27.4, 30.9), "lng_range": (74.5, 77.6)},
        "crime_rate_factor": 0.9
    },
    "Himachal Pradesh": {
        "major_cities": ["Shimla", "Dharamshala", "Solan", "Mandi", "Una", "Kullu", "Hamirpur"],
        "coordinates": {"lat_range": (30.2, 33.2), "lng_range": (75.6, 79.0)},
        "crime_rate_factor": 0.2
    },
    "Jharkhand": {
        "major_cities": ["Ranchi", "Jamshedpur", "Dhanbad", "Bokaro", "Deoghar", "Phusro", "Hazaribagh"],
        "coordinates": {"lat_range": (21.9, 25.3), "lng_range": (83.3, 87.9)},
        "crime_rate_factor": 0.6
    },
    "Karnataka": {
        "major_cities": ["Bangalore", "Mysore", "Hubli", "Mangalore", "Belgaum", "Gulbarga", "Davangere", "Bellary"],
        "coordinates": {"lat_range": (11.3, 18.4), "lng_range": (74.0, 78.6)},
        "crime_rate_factor": 0.8
    },
    "Kerala": {
        "major_cities": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kollam", "Alappuzha", "Palakkad"],
        "coordinates": {"lat_range": (8.2, 12.8), "lng_range": (74.8, 77.4)},
        "crime_rate_factor": 0.5
    },
    "Madhya Pradesh": {
        "major_cities": ["Indore", "Bhopal", "Jabalpur", "Gwalior", "Ujjain", "Sagar", "Dewas", "Satna"],
        "coordinates": {"lat_range": (21.0, 26.9), "lng_range": (74.0, 82.8)},
        "crime_rate_factor": 0.7
    },
    "Maharashtra": {
        "major_cities": ["Mumbai", "Pune", "Nagpur", "Thane", "Nashik", "Aurangabad", "Solapur", "Amravati"],
        "coordinates": {"lat_range": (15.6, 22.0), "lng_range": (72.6, 80.9)},
        "crime_rate_factor": 1.0
    },
    "Manipur": {
        "major_cities": ["Imphal", "Thoubal", "Kakching", "Churachandpur"],
        "coordinates": {"lat_range": (23.8, 25.7), "lng_range": (93.0, 94.8)},
        "crime_rate_factor": 0.3
    },
    "Meghalaya": {
        "major_cities": ["Shillong", "Tura", "Cherrapunji", "Jowai"],
        "coordinates": {"lat_range": (25.0, 26.1), "lng_range": (89.4, 92.8)},
        "crime_rate_factor": 0.2
    },
    "Mizoram": {
        "major_cities": ["Aizawl", "Lunglei", "Saiha", "Champhai"],
        "coordinates": {"lat_range": (21.9, 24.6), "lng_range": (92.2, 93.4)},
        "crime_rate_factor": 0.1
    },
    "Nagaland": {
        "major_cities": ["Kohima", "Dimapur", "Mokokchung", "Tuensang"],
        "coordinates": {"lat_range": (25.2, 27.0), "lng_range": (93.3, 95.8)},
        "crime_rate_factor": 0.2
    },
    "Odisha": {
        "major_cities": ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur", "Sambalpur", "Puri", "Balasore"],
        "coordinates": {"lat_range": (17.8, 22.6), "lng_range": (81.3, 87.5)},
        "crime_rate_factor": 0.5
    },
    "Punjab": {
        "major_cities": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda", "Mohali", "Pathankot"],
        "coordinates": {"lat_range": (29.4, 32.5), "lng_range": (73.9, 76.9)},
        "crime_rate_factor": 0.7
    },
    "Rajasthan": {
        "major_cities": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer", "Bikaner", "Alwar", "Bharatpur"],
        "coordinates": {"lat_range": (23.0, 30.2), "lng_range": (69.5, 78.2)},
        "crime_rate_factor": 0.8
    },
    "Sikkim": {
        "major_cities": ["Gangtok", "Namchi", "Geyzing", "Mangan"],
        "coordinates": {"lat_range": (27.0, 28.1), "lng_range": (88.0, 88.9)},
        "crime_rate_factor": 0.1
    },
    "Tamil Nadu": {
        "major_cities": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli", "Erode", "Vellore"],
        "coordinates": {"lat_range": (8.1, 13.6), "lng_range": (76.2, 80.3)},
        "crime_rate_factor": 0.9
    },
    "Telangana": {
        "major_cities": ["Hyderabad", "Warangal", "Nizamabad", "Khammam", "Karimnagar", "Mahbubnagar"],
        "coordinates": {"lat_range": (15.8, 19.9), "lng_range": (77.2, 81.8)},
        "crime_rate_factor": 0.9
    },
    "Tripura": {
        "major_cities": ["Agartala", "Dharmanagar", "Udaipur", "Kailashahar"],
        "coordinates": {"lat_range": (22.9, 24.5), "lng_range": (91.0, 92.7)},
        "crime_rate_factor": 0.3
    },
    "Uttar Pradesh": {
        "major_cities": ["Lucknow", "Kanpur", "Ghaziabad", "Agra", "Varanasi", "Meerut", "Allahabad", "Bareilly"],
        "coordinates": {"lat_range": (23.9, 30.4), "lng_range": (77.1, 84.6)},
        "crime_rate_factor": 1.0
    },
    "Uttarakhand": {
        "major_cities": ["Dehradun", "Haridwar", "Roorkee", "Rudrapur", "Kashipur", "Rishikesh"],
        "coordinates": {"lat_range": (28.4, 31.4), "lng_range": (77.6, 81.0)},
        "crime_rate_factor": 0.4
    },
    "West Bengal": {
        "major_cities": ["Kolkata", "Howrah", "Durgapur", "Asansol", "Siliguri", "Malda", "Bardhaman"],
        "coordinates": {"lat_range": (21.8, 27.1), "lng_range": (85.8, 89.9)},
        "crime_rate_factor": 0.8
    },
    "Delhi": {
        "major_cities": ["New Delhi", "Delhi", "Dwarka", "Rohini", "Janakpuri", "Saket", "Lajpat Nagar"],
        "coordinates": {"lat_range": (28.4, 28.9), "lng_range": (76.8, 77.3)},
        "crime_rate_factor": 1.0
    }
}

# Crime types with Indian context
CRIME_TYPES = {
    "Digital Payment Fraud": {"weight": 0.35, "avg_amount": (50000, 500000)},
    "UPI Fraud": {"weight": 0.25, "avg_amount": (5000, 100000)},
    "Credit Card Fraud": {"weight": 0.15, "avg_amount": (20000, 300000)},
    "Phishing": {"weight": 0.10, "avg_amount": (10000, 150000)},
    "Cyberstalking": {"weight": 0.05, "avg_amount": (0, 25000)},
    "Online Job Fraud": {"weight": 0.05, "avg_amount": (15000, 200000)},
    "Social Media Fraud": {"weight": 0.03, "avg_amount": (5000, 75000)},
    "Cryptocurrency Fraud": {"weight": 0.02, "avg_amount": (100000, 2000000)}
}

class EnhancedIndianCybercrimeGenerator:
    def __init__(self):
        self.base_date = datetime.now() - timedelta(days=365)
        
    def generate_realistic_coordinates(self, state_info: Dict) -> tuple:
        """Generate realistic coordinates within state boundaries"""
        lat_range = state_info["coordinates"]["lat_range"]
        lng_range = state_info["coordinates"]["lng_range"]
        
        lat = random.uniform(lat_range[0], lat_range[1])
        lng = random.uniform(lng_range[0], lng_range[1])
        
        return round(lat, 6), round(lng, 6)
    
    def generate_indian_phone(self) -> str:
        """Generate realistic Indian phone numbers"""
        prefixes = ["98", "99", "97", "96", "95", "94", "93", "92", "91", "90", "89", "88", "87", "86", "85", "84", "83", "82", "81", "80"]
        prefix = random.choice(prefixes)
        number = ''.join([str(random.randint(0, 9)) for _ in range(8)])
        return f"+91-{prefix}{number}"
    
    def generate_crime_pattern_by_time(self, hour: int) -> str:
        """Generate crime patterns based on time of day"""
        if 0 <= hour <= 6:  # Late night/Early morning
            return random.choice(["UPI Fraud", "Digital Payment Fraud", "Cryptocurrency Fraud"])
        elif 7 <= hour <= 11:  # Morning
            return random.choice(["Online Job Fraud", "Phishing", "Social Media Fraud"])
        elif 12 <= hour <= 17:  # Afternoon
            return random.choice(["Credit Card Fraud", "Digital Payment Fraud", "UPI Fraud"])
        else:  # Evening
            return random.choice(["Cyberstalking", "Social Media Fraud", "Phishing"])
    
    def generate_seasonal_pattern(self, date: datetime) -> float:
        """Generate seasonal crime rate multipliers"""
        month = date.month
        if month in [11, 12, 1]:  # Festive season - higher online activity
            return 1.4
        elif month in [4, 5, 6]:  # Summer - moderate activity
            return 1.1
        elif month in [7, 8, 9]:  # Monsoon - lower activity
            return 0.8
        else:  # Spring - normal activity
            return 1.0
    
    def generate_comprehensive_dataset(self, num_records: int = 50000) -> List[Dict]:
        """Generate comprehensive India-wide cybercrime dataset"""
        dataset = []
        
        print(f"🇮🇳 Generating {num_records} records covering all Indian states...")
        
        for i in range(num_records):
            # Select random state and city
            state = random.choice(list(INDIAN_STATES_CITIES.keys()))
            state_info = INDIAN_STATES_CITIES[state]
            city = random.choice(state_info["major_cities"])
            
            # Generate timestamp with realistic patterns
            days_ago = random.randint(0, 365)
            base_time = self.base_date + timedelta(days=days_ago)
            
            # Add time patterns (more crimes during certain hours)
            hour_weights = [0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0,
                           2.2, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.7, 0.8, 0.7, 0.6]
            hour = random.choices(range(24), weights=hour_weights)[0]
            
            timestamp = base_time.replace(
                hour=hour,
                minute=random.randint(0, 59),
                second=random.randint(0, 59)
            )
            
            # Crime type based on time and location
            crime_type = self.generate_crime_pattern_by_time(hour)
            if random.random() < 0.3:  # 30% random selection
                crime_type = random.choices(
                    list(CRIME_TYPES.keys()),
                    weights=[CRIME_TYPES[ct]["weight"] for ct in CRIME_TYPES.keys()]
                )[0]
            
            # Generate coordinates
            complaint_lat, complaint_lng = self.generate_realistic_coordinates(state_info)
            
            # Amount lost with realistic patterns
            amount_range = CRIME_TYPES[crime_type]["avg_amount"]
            base_amount = random.uniform(amount_range[0], amount_range[1])
            
            # Apply regional multipliers
            regional_multiplier = 1.0
            if state in ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Telangana"]:
                regional_multiplier = 1.5  # Tech hubs - higher amounts
            elif state in ["Uttar Pradesh", "Bihar", "West Bengal"]:
                regional_multiplier = 0.8  # Lower average income areas
            
            # Apply seasonal patterns
            seasonal_multiplier = self.generate_seasonal_pattern(timestamp)
            
            amount_lost = round(base_amount * regional_multiplier * seasonal_multiplier, 2)
            
            # Urgency based on amount and crime type
            if amount_lost > 200000 or crime_type in ["Cryptocurrency Fraud", "Digital Payment Fraud"]:
                urgency = "High"
            elif amount_lost > 50000:
                urgency = "Medium"
            else:
                urgency = "Low"
            
            # Generate withdrawal prediction (offset from complaint location)
            offset_km = random.uniform(2, 25)  # 2-25 km offset
            lat_offset = (offset_km / 111.0) * random.choice([-1, 1])  # 1 degree ≈ 111 km
            lng_offset = (offset_km / (111.0 * abs(complaint_lat / 90))) * random.choice([-1, 1])
            
            withdrawal_lat = round(complaint_lat + lat_offset, 6)
            withdrawal_lng = round(complaint_lng + lng_offset, 6)
            
            # Risk calculation
            risk_factors = []
            risk_factors.append(state_info["crime_rate_factor"])
            risk_factors.append(1.2 if hour in [22, 23, 0, 1, 2, 3] else 0.8)  # Time risk
            risk_factors.append(1.3 if amount_lost > 100000 else 0.9)  # Amount risk
            risk_factors.append(1.1 if crime_type in ["Digital Payment Fraud", "UPI Fraud"] else 0.95)
            
            risk_score = min(0.95, sum(risk_factors) / len(risk_factors))
            
            record = {
                "complaint_id": f"CG{timestamp.year}{i+1:06d}",
                "timestamp": timestamp.isoformat(),
                "crime_type": crime_type,
                "amount_lost": amount_lost,
                "urgency_level": urgency,
                "complaint_city": city,
                "complaint_state": state,
                "complaint_lat": complaint_lat,
                "complaint_lng": complaint_lng,
                "victim_phone": self.generate_indian_phone(),
                "predicted_withdrawal_lat": withdrawal_lat,
                "predicted_withdrawal_lng": withdrawal_lng,
                "withdrawal_probability": round(risk_score, 3),
                "risk_score": round(risk_score * 100, 1),
                "status": random.choice(["Under Investigation", "Resolved", "Pending", "Closed"]),
                "reported_by": random.choice(["Online Portal", "Phone Call", "Police Station", "Bank"]),
                "investigation_officer": f"Officer_{random.randint(1000, 9999)}",
                "bank_involved": random.choice([
                    "State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank", 
                    "Punjab National Bank", "Bank of Baroda", "Canara Bank", "IDFC Bank"
                ])
            }
            
            dataset.append(record)
            
            if (i + 1) % 5000 == 0:
                print(f"✅ Generated {i + 1} records...")
        
        print(f"🎯 Dataset generation complete! {len(dataset)} records created")
        return dataset
    
    def save_enhanced_dataset(self, dataset: List[Dict], filename: str = "enhanced_indian_cybercrime_data.json"):
        """Save the enhanced dataset to file"""
        # Save as JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for analysis
        df = pd.DataFrame(dataset)
        csv_filename = filename.replace('.json', '.csv')
        df.to_csv(csv_filename, index=False)
        
        print(f"💾 Dataset saved as {filename} and {csv_filename}")
        return filename, csv_filename
    
    def generate_summary_statistics(self, dataset: List[Dict]) -> Dict:
        """Generate summary statistics for the dataset"""
        df = pd.DataFrame(dataset)
        
        stats = {
            "total_records": len(dataset),
            "states_covered": df['complaint_state'].nunique(),
            "cities_covered": df['complaint_city'].nunique(),
            "crime_types": df['crime_type'].value_counts().to_dict(),
            "state_wise_distribution": df['complaint_state'].value_counts().to_dict(),
            "total_amount_lost": df['amount_lost'].sum(),
            "average_amount_lost": df['amount_lost'].mean(),
            "high_risk_cases": len(df[df['risk_score'] > 75]),
            "urgency_distribution": df['urgency_level'].value_counts().to_dict(),
            "monthly_trend": df.groupby(pd.to_datetime(df['timestamp']).dt.month)['complaint_id'].count().to_dict()
        }
        
        return stats

if __name__ == "__main__":
    generator = EnhancedIndianCybercrimeGenerator()
    
    # Generate comprehensive dataset
    dataset = generator.generate_comprehensive_dataset(50000)
    
    # Save dataset
    json_file, csv_file = generator.save_enhanced_dataset(dataset)
    
    # Generate summary
    stats = generator.generate_summary_statistics(dataset)
    print("\n📊 Dataset Summary:")
    print(f"Total Records: {stats['total_records']:,}")
    print(f"States Covered: {stats['states_covered']}")
    print(f"Cities Covered: {stats['cities_covered']}")
    print(f"Total Amount Lost: ₹{stats['total_amount_lost']:,.2f}")
    print(f"High Risk Cases: {stats['high_risk_cases']:,}")
    
    # Save summary
    with open('dataset_summary.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n🚀 Enhanced Indian cybercrime dataset ready for CyberGuard system!")