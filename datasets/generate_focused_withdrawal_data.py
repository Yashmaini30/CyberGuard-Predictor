#!/usr/bin/env python3
"""
Focused Cash Withdrawal Prediction Dataset Generator
Specifically designed for SIH Problem Statement ID: 25257

Creates datasets optimized for:
1. Predictive Analytics Engine - Cash withdrawal hotspot prediction
2. Risk Heatmap Dashboard - Geospatial risk visualization
3. Law Enforcement Interface - Alert and intelligence data
4. Alert & Notification System - Real-time notification triggers
"""

import csv
import json
import random
from datetime import datetime, timedelta
import os

class WithdrawalPredictionDatasetGenerator:
    def __init__(self):
        self.output_dir = "datasets/withdrawal_prediction"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Focus on financial fraud crimes that lead to cash withdrawals
        self.withdrawal_relevant_crimes = [
            "UPI_Fraud", "Credit_Card_Fraud", "Debit_Card_Fraud", 
            "Banking_Fraud", "ATM_Fraud", "Net_Banking_Fraud",
            "Mobile_Banking_Fraud", "OTP_Fraud"
        ]
        
        # High-risk withdrawal locations in major Indian cities
        self.withdrawal_hotspots = {
            "Mumbai": [(19.0760, 72.8777), (19.0896, 72.8656), (19.1136, 72.8697)],
            "Delhi": [(28.7041, 77.1025), (28.6139, 77.2090), (28.5355, 77.3910)],
            "Bangalore": [(12.9716, 77.5946), (12.9719, 77.5937), (12.9698, 77.7500)],
            "Hyderabad": [(17.3850, 78.4867), (17.4065, 78.4772), (17.4399, 78.3489)],
            "Chennai": [(13.0827, 80.2707), (13.0878, 80.2785), (13.0569, 80.2421)]
        }
        
        # ATM networks and banking locations
        self.atm_networks = ["SBI", "HDFC", "ICICI", "Axis", "PNB", "BOB", "Canara"]
        
    def generate_withdrawal_prediction_dataset(self, num_records=8000):
        """Generate focused dataset for withdrawal location prediction"""
        print(f"🎯 Generating {num_records} withdrawal prediction records...")
        
        records = []
        
        for i in range(num_records):
            if i % 1000 == 0:
                print(f"Generated {i} records...")
                
            # Focus on withdrawal-relevant crimes only
            crime_type = random.choice(self.withdrawal_relevant_crimes)
            
            # Select city and complaint location
            city = random.choice(list(self.withdrawal_hotspots.keys()))
            complaint_lat, complaint_lng = self.generate_city_coordinates(city)
            
            # Generate temporal features (8000 complaints daily pattern)
            timestamp = self.generate_realistic_timestamp()
            
            # Generate complaint details focused on cash withdrawal potential
            amount_lost = self.generate_withdrawal_relevant_amount(crime_type)
            
            # Predict withdrawal location based on crime pattern
            withdrawal_lat, withdrawal_lng, withdrawal_probability = self.predict_withdrawal_location(
                complaint_lat, complaint_lng, crime_type, amount_lost, city
            )
            
            # Calculate time to withdrawal (critical for intervention)
            hours_to_withdrawal = self.estimate_withdrawal_timing(crime_type, amount_lost)
            
            # Generate ATM/bank details for the predicted location
            nearest_atm = self.find_nearest_atm(withdrawal_lat, withdrawal_lng)
            
            record = {
                # Complaint Identification
                "complaint_id": f"NCRP{random.randint(100000000, 999999999)}",
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "crime_type": crime_type,
                "amount_lost": amount_lost,
                "urgency_level": self.calculate_urgency(amount_lost, crime_type),
                
                # Geographic Features
                "complaint_city": city,
                "complaint_lat": round(complaint_lat, 6),
                "complaint_lng": round(complaint_lng, 6),
                
                # Predicted Withdrawal Location (TARGET)
                "predicted_withdrawal_lat": round(withdrawal_lat, 6),
                "predicted_withdrawal_lng": round(withdrawal_lng, 6),
                "withdrawal_probability": round(withdrawal_probability, 3),
                "hours_to_withdrawal": hours_to_withdrawal,
                
                # ATM/Banking Infrastructure
                "nearest_atm_network": nearest_atm["network"],
                "nearest_atm_lat": round(nearest_atm["lat"], 6),
                "nearest_atm_lng": round(nearest_atm["lng"], 6),
                "atm_distance_km": round(nearest_atm["distance"], 2),
                
                # Risk Assessment
                "risk_score": self.calculate_risk_score(
                    amount_lost, crime_type, withdrawal_probability, hours_to_withdrawal
                ),
                "intervention_window_hours": max(1, hours_to_withdrawal - 2),
                
                # Temporal Features
                "hour": timestamp.hour,
                "day_of_week": timestamp.weekday(),
                "is_weekend": 1 if timestamp.weekday() >= 5 else 0,
                "is_peak_withdrawal_time": 1 if timestamp.hour in [10, 11, 14, 15, 19, 20] else 0,
                
                # LEA Coordination
                "jurisdiction": self.get_jurisdiction(city),
                "alert_priority": self.get_alert_priority(withdrawal_probability, amount_lost),
                "requires_bank_alert": 1 if withdrawal_probability > 0.7 else 0
            }
            
            records.append(record)
        
        # Save the focused dataset
        filename = f"{self.output_dir}/complaint_to_withdrawal_focused.csv"
        with open(filename, "w", newline="") as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
        
        print(f"✅ Withdrawal prediction dataset created: {filename}")
        return records
        
    def generate_city_coordinates(self, city):
        """Generate realistic coordinates within city bounds"""
        base_coords = self.withdrawal_hotspots[city][0]
        
        # Add variance within city limits (~20km radius)
        lat_variance = random.uniform(-0.15, 0.15)
        lng_variance = random.uniform(-0.15, 0.15)
        
        return base_coords[0] + lat_variance, base_coords[1] + lng_variance
        
    def generate_realistic_timestamp(self):
        """Generate timestamp following 8000 daily complaints pattern"""
        # Random date in last 6 months (recent pattern)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        random_date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        # Peak complaint hours: 9-11 AM, 2-4 PM, 7-9 PM
        peak_hours = [9, 10, 11, 14, 15, 16, 19, 20, 21]
        if random.random() < 0.6:  # 60% during peak hours
            hour = random.choice(peak_hours)
            random_date = random_date.replace(hour=hour)
            
        return random_date
        
    def generate_withdrawal_relevant_amount(self, crime_type):
        """Generate amounts that typically lead to cash withdrawals"""
        # Amounts that criminals typically try to withdraw quickly
        withdrawal_amounts = {
            "UPI_Fraud": (5000, 50000),
            "Credit_Card_Fraud": (10000, 200000),
            "Debit_Card_Fraud": (5000, 100000),
            "Banking_Fraud": (25000, 500000),
            "ATM_Fraud": (2000, 40000),
            "Net_Banking_Fraud": (15000, 300000),
            "Mobile_Banking_Fraud": (5000, 75000),
            "OTP_Fraud": (3000, 30000)
        }
        
        min_amt, max_amt = withdrawal_amounts.get(crime_type, (5000, 50000))
        return round(random.uniform(min_amt, max_amt), 2)
        
    def predict_withdrawal_location(self, complaint_lat, complaint_lng, crime_type, amount, city):
        """Predict likely withdrawal location based on crime patterns"""
        
        # High-value crimes -> farther from complaint location (avoid detection)
        if amount > 100000:
            distance_factor = random.uniform(3, 8)  # 3-8 km away
            probability = random.uniform(0.75, 0.95)  # High probability
        elif amount > 50000:
            distance_factor = random.uniform(1, 5)  # 1-5 km away  
            probability = random.uniform(0.6, 0.8)   # Medium-high probability
        else:
            distance_factor = random.uniform(0.5, 3)  # 0.5-3 km away
            probability = random.uniform(0.4, 0.7)   # Medium probability
            
        # Convert distance to lat/lng offset
        lat_offset = (distance_factor / 111.0) * random.choice([-1, 1])
        lng_offset = (distance_factor / 111.0) * random.choice([-1, 1])
        
        withdrawal_lat = complaint_lat + lat_offset
        withdrawal_lng = complaint_lng + lng_offset
        
        # Ensure coordinates stay within city bounds
        city_hotspots = self.withdrawal_hotspots[city]
        if len(city_hotspots) > 1:
            # Bias toward known hotspots
            if random.random() < 0.3:  # 30% chance of hotspot
                hotspot = random.choice(city_hotspots[1:])  # Exclude base coordinates
                withdrawal_lat = hotspot[0] + random.uniform(-0.02, 0.02)
                withdrawal_lng = hotspot[1] + random.uniform(-0.02, 0.02)
                probability = min(probability + 0.1, 0.95)
        
        return withdrawal_lat, withdrawal_lng, probability
        
    def estimate_withdrawal_timing(self, crime_type, amount):
        """Estimate hours until withdrawal attempt"""
        # Urgent crimes (OTP, ATM fraud) -> immediate withdrawal
        if crime_type in ["OTP_Fraud", "ATM_Fraud"]:
            return random.randint(1, 6)  # 1-6 hours
        
        # High amounts -> quick withdrawal before blocking
        if amount > 100000:
            return random.randint(2, 12)  # 2-12 hours
        elif amount > 50000:
            return random.randint(6, 24)  # 6-24 hours
        else:
            return random.randint(12, 72)  # 12-72 hours
            
    def find_nearest_atm(self, lat, lng):
        """Find nearest ATM for withdrawal location"""
        network = random.choice(self.atm_networks)
        
        # Generate ATM location near withdrawal point
        atm_lat = lat + random.uniform(-0.01, 0.01)  # Within 1km
        atm_lng = lng + random.uniform(-0.01, 0.01)
        
        # Calculate distance
        distance = ((lat - atm_lat) ** 2 + (lng - atm_lng) ** 2) ** 0.5 * 111  # Approx km
        
        return {
            "network": network,
            "lat": atm_lat,
            "lng": atm_lng,
            "distance": distance
        }
        
    def calculate_risk_score(self, amount, crime_type, withdrawal_prob, hours_to_withdrawal):
        """Calculate overall risk score for prioritization"""
        risk = 0.2  # Base risk
        
        # Amount-based risk
        if amount > 200000:
            risk += 0.3
        elif amount > 100000:
            risk += 0.2
        elif amount > 50000:
            risk += 0.1
            
        # Crime type risk
        urgent_crimes = ["OTP_Fraud", "ATM_Fraud", "Debit_Card_Fraud"]
        if crime_type in urgent_crimes:
            risk += 0.2
            
        # Withdrawal probability
        risk += withdrawal_prob * 0.3
        
        # Time urgency
        if hours_to_withdrawal <= 6:
            risk += 0.2
        elif hours_to_withdrawal <= 24:
            risk += 0.1
            
        return min(risk, 1.0)
        
    def calculate_urgency(self, amount, crime_type):
        """Calculate urgency level for triage"""
        if amount > 500000 or crime_type in ["ATM_Fraud", "OTP_Fraud"]:
            return "CRITICAL"
        elif amount > 100000:
            return "HIGH"
        elif amount > 50000:
            return "MEDIUM"
        else:
            return "LOW"
            
    def get_jurisdiction(self, city):
        """Get LEA jurisdiction for coordination"""
        jurisdiction_map = {
            "Mumbai": "Maharashtra Police",
            "Delhi": "Delhi Police", 
            "Bangalore": "Karnataka Police",
            "Hyderabad": "Telangana Police",
            "Chennai": "Tamil Nadu Police"
        }
        return jurisdiction_map.get(city, "State Police")
        
    def get_alert_priority(self, withdrawal_prob, amount):
        """Determine alert priority for LEAs and banks"""
        if withdrawal_prob > 0.8 and amount > 100000:
            return "P1"  # Immediate action
        elif withdrawal_prob > 0.6 and amount > 50000:
            return "P2"  # High priority
        elif withdrawal_prob > 0.4:
            return "P3"  # Standard priority
        else:
            return "P4"  # Monitor only

    def create_geospatial_risk_heatmap_data(self):
        """Generate data specifically for risk heatmap dashboard"""
        print("🗺️ Creating geospatial risk heatmap data...")
        
        heatmap_data = []
        
        for city in self.withdrawal_hotspots.keys():
            # Create grid points across city for heatmap
            base_lat, base_lng = self.withdrawal_hotspots[city][0]
            
            for lat_offset in [-0.1, -0.05, 0, 0.05, 0.1]:
                for lng_offset in [-0.1, -0.05, 0, 0.05, 0.1]:
                    grid_lat = base_lat + lat_offset
                    grid_lng = base_lng + lng_offset
                    
                    # Calculate risk intensity for this grid point
                    risk_intensity = random.uniform(0.1, 0.9)
                    incident_count = random.randint(1, 50)
                    
                    heatmap_data.append({
                        "lat": round(grid_lat, 6),
                        "lng": round(grid_lng, 6),
                        "city": city,
                        "risk_intensity": round(risk_intensity, 3),
                        "incident_count_30days": incident_count,
                        "avg_amount": random.randint(25000, 200000),
                        "hotspot_category": self.categorize_hotspot(risk_intensity),
                        "last_incident": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
                    })
        
        # Save heatmap data
        with open(f"{self.output_dir}/risk_heatmap_data.csv", "w", newline="") as f:
            if heatmap_data:
                writer = csv.DictWriter(f, fieldnames=heatmap_data[0].keys())
                writer.writeheader()
                writer.writerows(heatmap_data)
        
        print(f"✅ Risk heatmap data created: {len(heatmap_data)} grid points")
        return heatmap_data
        
    def categorize_hotspot(self, risk_intensity):
        """Categorize risk level for visualization"""
        if risk_intensity > 0.7:
            return "CRITICAL"
        elif risk_intensity > 0.5:
            return "HIGH"
        elif risk_intensity > 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def create_alert_triggers_dataset(self):
        """Generate alert trigger patterns for notification system"""
        print("🚨 Creating alert triggers dataset...")
        
        alert_data = []
        
        # Generate alert scenarios based on real-time patterns
        for _ in range(1000):
            trigger_time = datetime.now() - timedelta(hours=random.randint(1, 720))  # Last 30 days
            
            alert_data.append({
                "alert_id": f"ALT{random.randint(100000, 999999)}",
                "trigger_timestamp": trigger_time.strftime("%Y-%m-%d %H:%M:%S"),
                "alert_type": random.choice(["WITHDRAWAL_IMMINENT", "HOTSPOT_ACTIVITY", "PATTERN_MATCH"]),
                "priority": random.choice(["P1", "P2", "P3", "P4"]),
                "predicted_location_lat": round(random.uniform(12.9, 28.7), 6),
                "predicted_location_lng": round(random.uniform(72.8, 80.3), 6),
                "confidence_score": round(random.uniform(0.4, 0.95), 3),
                "estimated_withdrawal_time": (trigger_time + timedelta(hours=random.randint(1, 24))).strftime("%Y-%m-%d %H:%M:%S"),
                "amount_at_risk": random.randint(10000, 500000),
                "lea_notified": random.choice([True, False]),
                "bank_notified": random.choice([True, False]),
                "response_time_minutes": random.randint(5, 120),
                "intervention_success": random.choice([True, False, None])  # None = pending
            })
        
        # Save alert data
        with open(f"{self.output_dir}/alert_triggers.csv", "w", newline="") as f:
            if alert_data:
                writer = csv.DictWriter(f, fieldnames=alert_data[0].keys())
                writer.writeheader()
                writer.writerows(alert_data)
        
        print(f"✅ Alert triggers dataset created: {len(alert_data)} alerts")
        return alert_data

    def create_focused_documentation(self):
        """Create documentation focused on problem statement deliverables"""
        docs = {
            "problem_statement": {
                "id": "25257",
                "title": "Predictive Analytics Framework for Cybercrime Cash Withdrawal Location Forecasting",
                "focus": "Predict likely cash withdrawal locations to enable proactive intervention"
            },
            "datasets_created": {
                "complaint_to_withdrawal_focused.csv": {
                    "purpose": "Primary training dataset for ML model",
                    "records": 8000,
                    "features": "Optimized for withdrawal location prediction",
                    "target_variables": ["predicted_withdrawal_lat", "predicted_withdrawal_lng", "withdrawal_probability"]
                },
                "risk_heatmap_data.csv": {
                    "purpose": "GIS-enabled dashboard visualization", 
                    "records": 125,
                    "features": "Grid-based risk intensity mapping",
                    "use_case": "Real-time risk zone visualization"
                },
                "alert_triggers.csv": {
                    "purpose": "Alert & notification system patterns",
                    "records": 1000,
                    "features": "Alert triggers and response tracking",
                    "use_case": "Real-time notification to LEAs and banks"
                },
                "state_risk_factors.csv": {
                    "purpose": "NCRB-validated state risk factors",
                    "records": 960,
                    "features": "Official government statistics",
                    "use_case": "Risk factor validation and adjustment"
                }
            },
            "key_deliverables_mapping": {
                "Predictive Analytics Engine": "complaint_to_withdrawal_focused.csv + state_risk_factors.csv",
                "Risk Heatmap Dashboard": "risk_heatmap_data.csv",
                "Law Enforcement Interface": "alert_triggers.csv + complaint_to_withdrawal_focused.csv",
                "Alert & Notification System": "alert_triggers.csv"
            },
            "optimization_focus": [
                "Cash withdrawal location prediction accuracy",
                "Real-time intervention window calculation", 
                "Geospatial risk modeling",
                "LEA and bank coordination",
                "8000 daily complaints processing capability"
            ]
        }
        
        with open(f"{self.output_dir}/focused_dataset_documentation.json", "w") as f:
            json.dump(docs, f, indent=2)
        
        print("✅ Focused documentation created")
        return docs

def main():
    """Generate focused datasets for withdrawal prediction problem statement"""
    print("🎯 Focused Dataset Generation for Cash Withdrawal Prediction")
    print("📋 SIH Problem Statement ID: 25257")
    print("🎯 Focus: Predict cash withdrawal locations for proactive intervention")
    print("-" * 70)
    
    generator = WithdrawalPredictionDatasetGenerator()
    
    # Generate focused datasets
    generator.generate_withdrawal_prediction_dataset(8000)  # Match daily complaint volume
    generator.create_geospatial_risk_heatmap_data()
    generator.create_alert_triggers_dataset()
    generator.create_focused_documentation()
    
    print("\n🎯 Focused Dataset Generation Complete!")
    print("📁 Datasets optimized for:")
    print("   🤖 Predictive Analytics Engine")
    print("   🗺️ Risk Heatmap Dashboard") 
    print("   👮 Law Enforcement Interface")
    print("   🚨 Alert & Notification System")
    print("\n🏆 Ready for SIH cash withdrawal prediction demo!")

if __name__ == "__main__":
    main()