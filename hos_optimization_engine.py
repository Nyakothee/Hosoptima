"""
Fleet Optimization Engine for HOS Violation Prediction System
Intelligent scheduling, route optimization, and resource allocation
Production-ready with genetic algorithms, constraint programming, and ML-guided optimization
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
import json
from enum import Enum
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries
try:
    from scipy.optimize import linear_sum_assignment, minimize
    from scipy.spatial import distance_matrix
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Install with: pip install scipy")

try:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    logging.warning("OR-Tools not available. Install with: pip install ortools")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives"""
    MINIMIZE_VIOLATIONS = "minimize_violations"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCE_WORKLOAD = "balance_workload"


@dataclass
class Driver:
    """Driver information for optimization"""
    driver_id: str
    name: str
    experience_years: float
    violation_history: int
    current_hours_worked_week: float
    max_daily_hours: float = 11.0
    max_weekly_hours: float = 60.0
    available: bool = True
    location: Tuple[float, float] = (0.0, 0.0)  # lat, lon
    predicted_violation_risk: float = 0.0
    
    def hours_remaining_daily(self, hours_today: float = 0) -> float:
        """Calculate remaining daily hours"""
        return max(0, self.max_daily_hours - hours_today)
    
    def hours_remaining_weekly(self) -> float:
        """Calculate remaining weekly hours"""
        return max(0, self.max_weekly_hours - self.current_hours_worked_week)


@dataclass
class Route:
    """Route information"""
    route_id: str
    origin: Tuple[float, float]
    destination: Tuple[float, float]
    estimated_hours: float
    priority: int  # 1-10, 10 being highest
    required_breaks: int = 0
    distance_miles: float = 0.0
    delivery_time: datetime = None
    
    def calculate_breaks_required(self) -> int:
        """Calculate required breaks for route"""
        return int(self.estimated_hours // 8)  # Break every 8 hours


@dataclass
class Assignment:
    """Driver-route assignment"""
    driver: Driver
    route: Route
    start_time: datetime
    estimated_completion: datetime
    violation_risk_score: float
    cost: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'driver_id': self.driver.driver_id,
            'driver_name': self.driver.name,
            'route_id': self.route.route_id,
            'start_time': self.start_time.isoformat(),
            'estimated_completion': self.estimated_completion.isoformat(),
            'estimated_hours': self.route.estimated_hours,
            'violation_risk_score': self.violation_risk_score,
            'cost': self.cost
        }


@dataclass
class Schedule:
    """Complete fleet schedule"""
    schedule_id: str
    date: datetime
    assignments: List[Assignment]
    total_cost: float = 0.0
    total_violation_risk: float = 0.0
    unassigned_routes: List[Route] = field(default_factory=list)
    
    def calculate_metrics(self):
        """Calculate schedule metrics"""
        self.total_cost = sum(a.cost for a in self.assignments)
        self.total_violation_risk = sum(a.violation_risk_score for a in self.assignments)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'schedule_id': self.schedule_id,
            'date': self.date.isoformat(),
            'total_assignments': len(self.assignments),
            'assignments': [a.to_dict() for a in self.assignments],
            'total_cost': self.total_cost,
            'total_violation_risk': self.total_violation_risk,
            'unassigned_routes': [r.route_id for r in self.unassigned_routes]
        }


class ViolationRiskCalculator:
    """Calculate violation risk for driver-route assignments"""
    
    def __init__(self, predictor=None):
        self.predictor = predictor
    
    def calculate_risk(self, driver: Driver, route: Route, 
                      start_time: datetime) -> float:
        """Calculate violation risk for assignment"""
        risk_score = 0.0
        
        # Factor 1: Hours remaining vs route duration
        hours_needed = route.estimated_hours
        daily_remaining = driver.hours_remaining_daily()
        weekly_remaining = driver.hours_remaining_weekly()
        
        if hours_needed > daily_remaining:
            risk_score += 0.5  # Will exceed daily limit
        elif hours_needed > daily_remaining * 0.9:
            risk_score += 0.3  # Close to daily limit
        
        if hours_needed > weekly_remaining:
            risk_score += 0.4  # Will exceed weekly limit
        elif hours_needed > weekly_remaining * 0.9:
            risk_score += 0.2  # Close to weekly limit
        
        # Factor 2: Driver violation history
        if driver.violation_history > 0:
            risk_score += min(0.2, driver.violation_history * 0.05)
        
        # Factor 3: Driver experience (less experience = higher risk)
        if driver.experience_years < 2:
            risk_score += 0.15
        elif driver.experience_years < 5:
            risk_score += 0.05
        
        # Factor 4: Break requirements
        breaks_required = route.calculate_breaks_required()
        if breaks_required > 0:
            risk_score += breaks_required * 0.05
        
        # Factor 5: Predicted risk from ML model
        if driver.predicted_violation_risk > 0:
            risk_score += driver.predicted_violation_risk * 0.3
        
        # Cap at 1.0
        return min(1.0, risk_score)
    
    def calculate_risk_with_ml(self, driver: Driver, route: Route,
                               historical_data: np.ndarray) -> float:
        """Calculate risk using ML predictor"""
        if self.predictor is None:
            return self.calculate_risk(driver, route, datetime.now())
        
        try:
            # Prepare features for ML model
            features = self._prepare_features(driver, route, historical_data)
            
            # Get prediction
            prediction = self.predictor.predict(features)
            
            # Extract violation probability
            violation_prob = np.max(prediction)
            
            return float(violation_prob)
            
        except Exception as e:
            logger.error(f"ML risk calculation failed: {str(e)}")
            return self.calculate_risk(driver, route, datetime.now())
    
    def _prepare_features(self, driver: Driver, route: Route,
                         historical_data: np.ndarray) -> np.ndarray:
        """Prepare features for ML model"""
        # This should match the feature engineering used in training
        features = np.array([
            route.estimated_hours,
            driver.current_hours_worked_week,
            driver.experience_years,
            driver.violation_history,
            route.calculate_breaks_required(),
            driver.hours_remaining_daily(),
            driver.hours_remaining_weekly(),
            # Add more features as needed
        ])
        
        return features.reshape(1, -1)


class GeneticAlgorithmOptimizer:
    """Genetic algorithm for fleet optimization"""
    
    def __init__(self, drivers: List[Driver], routes: List[Route],
                 risk_calculator: ViolationRiskCalculator,
                 objective: OptimizationObjective = OptimizationObjective.MINIMIZE_VIOLATIONS):
        self.drivers = drivers
        self.routes = routes
        self.risk_calculator = risk_calculator
        self.objective = objective
        
        self.population_size = 100
        self.generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
    
    def optimize(self) -> Schedule:
        """Run genetic algorithm optimization"""
        logger.info(f"Starting genetic algorithm optimization with {len(self.drivers)} drivers and {len(self.routes)} routes")
        
        # Initialize population
        population = self._initialize_population()
        
        best_schedule = None
        best_fitness = float('inf') if self.objective == OptimizationObjective.MINIMIZE_VIOLATIONS else float('-inf')
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(schedule) for schedule in population]
            
            # Track best
            current_best_idx = np.argmin(fitness_scores) if self.objective == OptimizationObjective.MINIMIZE_VIOLATIONS else np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            
            if (self.objective == OptimizationObjective.MINIMIZE_VIOLATIONS and current_best_fitness < best_fitness) or \
               (self.objective != OptimizationObjective.MINIMIZE_VIOLATIONS and current_best_fitness > best_fitness):
                best_fitness = current_best_fitness
                best_schedule = population[current_best_idx]
            
            # Selection
            selected = self._selection(population, fitness_scores)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    if random.random() < self.crossover_rate:
                        child1, child2 = self._crossover(selected[i], selected[i+1])
                    else:
                        child1, child2 = selected[i], selected[i+1]
                    
                    offspring.extend([child1, child2])
            
            # Mutation
            for schedule in offspring:
                if random.random() < self.mutation_rate:
                    self._mutate(schedule)
            
            population = offspring
            
            if (generation + 1) % 10 == 0:
                logger.info(f"Generation {generation + 1}/{self.generations}, Best Fitness: {best_fitness:.4f}")
        
        logger.info(f"Optimization complete. Final fitness: {best_fitness:.4f}")
        
        return best_schedule
    
    def _initialize_population(self) -> List[Schedule]:
        """Initialize random population of schedules"""
        population = []
        
        for _ in range(self.population_size):
            schedule = self._create_random_schedule()
            population.append(schedule)
        
        return population
    
    def _create_random_schedule(self) -> Schedule:
        """Create random schedule"""
        assignments = []
        used_drivers = set()
        unassigned_routes = []
        
        # Shuffle routes for random assignment
        shuffled_routes = random.sample(self.routes, len(self.routes))
        
        for route in shuffled_routes:
            # Find available driver
            available_drivers = [d for d in self.drivers 
                               if d.driver_id not in used_drivers 
                               and d.hours_remaining_daily() >= route.estimated_hours]
            
            if available_drivers:
                driver = random.choice(available_drivers)
                
                start_time = datetime.now()
                completion_time = start_time + timedelta(hours=route.estimated_hours)
                
                risk_score = self.risk_calculator.calculate_risk(
                    driver, route, start_time
                )
                
                assignment = Assignment(
                    driver=driver,
                    route=route,
                    start_time=start_time,
                    estimated_completion=completion_time,
                    violation_risk_score=risk_score,
                    cost=route.distance_miles * 2.5  # $2.5 per mile
                )
                
                assignments.append(assignment)
                used_drivers.add(driver.driver_id)
            else:
                unassigned_routes.append(route)
        
        schedule = Schedule(
            schedule_id=f"SCH_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            date=datetime.now(),
            assignments=assignments,
            unassigned_routes=unassigned_routes
        )
        
        schedule.calculate_metrics()
        
        return schedule
    
    def _evaluate_fitness(self, schedule: Schedule) -> float:
        """Evaluate fitness of schedule"""
        if self.objective == OptimizationObjective.MINIMIZE_VIOLATIONS:
            # Lower is better
            return schedule.total_violation_risk + len(schedule.unassigned_routes) * 0.5
        
        elif self.objective == OptimizationObjective.MINIMIZE_COST:
            return schedule.total_cost + len(schedule.unassigned_routes) * 1000
        
        elif self.objective == OptimizationObjective.MAXIMIZE_EFFICIENCY:
            # Higher is better (return negative for minimization)
            efficiency = len(schedule.assignments) / (schedule.total_cost + 1)
            return -efficiency
        
        elif self.objective == OptimizationObjective.BALANCE_WORKLOAD:
            # Calculate workload variance
            if not schedule.assignments:
                return float('inf')
            
            workloads = {}
            for assignment in schedule.assignments:
                driver_id = assignment.driver.driver_id
                workloads[driver_id] = workloads.get(driver_id, 0) + assignment.route.estimated_hours
            
            variance = np.var(list(workloads.values()))
            return variance
    
    def _selection(self, population: List[Schedule], 
                  fitness_scores: List[float]) -> List[Schedule]:
        """Tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
            
            if self.objective == OptimizationObjective.MINIMIZE_VIOLATIONS or \
               self.objective == OptimizationObjective.MINIMIZE_COST or \
               self.objective == OptimizationObjective.BALANCE_WORKLOAD:
                winner = min(tournament, key=lambda x: x[1])[0]
            else:
                winner = max(tournament, key=lambda x: x[1])[0]
            
            selected.append(winner)
        
        return selected
    
    def _crossover(self, parent1: Schedule, parent2: Schedule) -> Tuple[Schedule, Schedule]:
        """Single-point crossover"""
        # Simple crossover: swap assignments
        split_point = len(parent1.assignments) // 2
        
        child1_assignments = parent1.assignments[:split_point] + parent2.assignments[split_point:]
        child2_assignments = parent2.assignments[:split_point] + parent1.assignments[split_point:]
        
        child1 = Schedule(
            schedule_id=f"SCH_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            date=datetime.now(),
            assignments=child1_assignments
        )
        child1.calculate_metrics()
        
        child2 = Schedule(
            schedule_id=f"SCH_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            date=datetime.now(),
            assignments=child2_assignments
        )
        child2.calculate_metrics()
        
        return child1, child2
    
    def _mutate(self, schedule: Schedule):
        """Mutate schedule by swapping assignments"""
        if len(schedule.assignments) >= 2:
            idx1, idx2 = random.sample(range(len(schedule.assignments)), 2)
            
            # Swap drivers
            schedule.assignments[idx1].driver, schedule.assignments[idx2].driver = \
                schedule.assignments[idx2].driver, schedule.assignments[idx1].driver
            
            # Recalculate risk scores
            for assignment in [schedule.assignments[idx1], schedule.assignments[idx2]]:
                assignment.violation_risk_score = self.risk_calculator.calculate_risk(
                    assignment.driver,
                    assignment.route,
                    assignment.start_time
                )
            
            schedule.calculate_metrics()


class ConstraintProgrammingOptimizer:
    """Constraint programming approach using OR-Tools"""
    
    def __init__(self, drivers: List[Driver], routes: List[Route],
                 risk_calculator: ViolationRiskCalculator):
        if not ORTOOLS_AVAILABLE:
            raise RuntimeError("OR-Tools not available")
        
        self.drivers = drivers
        self.routes = routes
        self.risk_calculator = risk_calculator
    
    def optimize(self) -> Schedule:
        """Solve using constraint programming"""
        logger.info("Starting constraint programming optimization")
        
        # Create cost matrix (driver x route)
        cost_matrix = np.zeros((len(self.drivers), len(self.routes)))
        
        for i, driver in enumerate(self.drivers):
            for j, route in enumerate(self.routes):
                # Calculate cost (combination of risk and monetary cost)
                risk = self.risk_calculator.calculate_risk(driver, route, datetime.now())
                monetary_cost = route.distance_miles * 2.5
                
                # Penalize if driver can't complete route
                if driver.hours_remaining_daily() < route.estimated_hours:
                    cost_matrix[i, j] = 1000000  # Very high penalty
                else:
                    cost_matrix[i, j] = risk * 100 + monetary_cost
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create schedule
        assignments = []
        for driver_idx, route_idx in zip(row_ind, col_ind):
            if cost_matrix[driver_idx, route_idx] < 1000000:  # Valid assignment
                driver = self.drivers[driver_idx]
                route = self.routes[route_idx]
                
                start_time = datetime.now()
                completion_time = start_time + timedelta(hours=route.estimated_hours)
                
                assignment = Assignment(
                    driver=driver,
                    route=route,
                    start_time=start_time,
                    estimated_completion=completion_time,
                    violation_risk_score=self.risk_calculator.calculate_risk(driver, route, start_time),
                    cost=route.distance_miles * 2.5
                )
                
                assignments.append(assignment)
        
        # Identify unassigned routes
        assigned_route_indices = set(col_ind[row_ind < len(self.routes)])
        unassigned_routes = [self.routes[i] for i in range(len(self.routes)) 
                           if i not in assigned_route_indices]
        
        schedule = Schedule(
            schedule_id=f"SCH_CP_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            date=datetime.now(),
            assignments=assignments,
            unassigned_routes=unassigned_routes
        )
        
        schedule.calculate_metrics()
        
        logger.info(f"Optimization complete. Assigned {len(assignments)} routes, {len(unassigned_routes)} unassigned")
        
        return schedule


class AdvancedOptimizationEngine:
    """
    Complete optimization engine for fleet scheduling and route planning
    """
    
    def __init__(self, predictor=None):
        self.predictor = predictor
        self.risk_calculator = ViolationRiskCalculator(predictor)
        
        logger.info("Advanced Optimization Engine initialized")
    
    def optimize_fleet_schedule(self, drivers: List[Driver], routes: List[Route],
                                objective: OptimizationObjective = OptimizationObjective.MINIMIZE_VIOLATIONS,
                                method: str = 'genetic') -> Schedule:
        """Optimize fleet schedule"""
        
        logger.info(f"Optimizing schedule for {len(drivers)} drivers and {len(routes)} routes")
        logger.info(f"Objective: {objective.value}, Method: {method}")
        
        if method == 'genetic':
            optimizer = GeneticAlgorithmOptimizer(
                drivers, routes, self.risk_calculator, objective
            )
            schedule = optimizer.optimize()
        
        elif method == 'constraint':
            if not SCIPY_AVAILABLE:
                raise RuntimeError("SciPy not available for constraint programming")
            
            optimizer = ConstraintProgrammingOptimizer(
                drivers, routes, self.risk_calculator
            )
            schedule = optimizer.optimize()
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return schedule
    
    def recommend_best_driver(self, route: Route, available_drivers: List[Driver]) -> Optional[Driver]:
        """Recommend best driver for a specific route"""
        
        if not available_drivers:
            return None
        
        best_driver = None
        lowest_risk = float('inf')
        
        for driver in available_drivers:
            # Check if driver can complete route
            if driver.hours_remaining_daily() < route.estimated_hours:
                continue
            
            # Calculate risk
            risk = self.risk_calculator.calculate_risk(driver, route, datetime.now())
            
            if risk < lowest_risk:
                lowest_risk = risk
                best_driver = driver
        
        return best_driver
    
    def suggest_route_modifications(self, route: Route, driver: Driver) -> Dict:
        """Suggest modifications to reduce violation risk"""
        
        suggestions = {
            'original_risk': self.risk_calculator.calculate_risk(driver, route, datetime.now()),
            'modifications': []
        }
        
        # Check if breaks needed
        breaks_required = route.calculate_breaks_required()
        if breaks_required > 0:
            suggestions['modifications'].append({
                'type': 'add_breaks',
                'description': f'Add {breaks_required} mandatory break(s) of 30 minutes each',
                'impact': 'Reduces continuous driving violation risk'
            })
        
        # Check if route should be split
        if route.estimated_hours > driver.hours_remaining_daily():
            suggestions['modifications'].append({
                'type': 'split_route',
                'description': f'Split into 2 segments (driver has only {driver.hours_remaining_daily():.1f}h remaining)',
                'impact': 'Prevents daily hour limit violation'
            })
        
        # Check if delay start time helps
        current_week_hours = driver.current_hours_worked_week
        if current_week_hours + route.estimated_hours > driver.max_weekly_hours:
            hours_over = current_week_hours + route.estimated_hours - driver.max_weekly_hours
            suggestions['modifications'].append({
                'type': 'delay_start',
                'description': f'Delay start to next week (currently {hours_over:.1f}h over weekly limit)',
                'impact': 'Prevents weekly hour limit violation'
            })
        
        return suggestions
    
    def generate_optimization_report(self, schedule: Schedule) -> str:
        """Generate comprehensive optimization report"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FLEET OPTIMIZATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Schedule ID: {schedule.schedule_id}")
        report_lines.append(f"Date: {schedule.date.strftime('%Y-%m-%d')}")
        report_lines.append("")
        
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Assignments: {len(schedule.assignments)}")
        report_lines.append(f"Unassigned Routes: {len(schedule.unassigned_routes)}")
        report_lines.append(f"Total Cost: ${schedule.total_cost:,.2f}")
        report_lines.append(f"Total Violation Risk Score: {schedule.total_violation_risk:.2f}")
        report_lines.append(f"Average Risk per Assignment: {schedule.total_violation_risk / len(schedule.assignments) if schedule.assignments else 0:.3f}")
        report_lines.append("")
        
        # Risk distribution
        if schedule.assignments:
            risks = [a.violation_risk_score for a in schedule.assignments]
            report_lines.append("RISK DISTRIBUTION")
            report_lines.append("-" * 80)
            report_lines.append(f"Low Risk (< 0.3): {sum(1 for r in risks if r < 0.3)} assignments")
            report_lines.append(f"Medium Risk (0.3-0.6): {sum(1 for r in risks if 0.3 <= r < 0.6)} assignments")
            report_lines.append(f"High Risk (0.6-0.8): {sum(1 for r in risks if 0.6 <= r < 0.8)} assignments")
            report_lines.append(f"Critical Risk (>= 0.8): {sum(1 for r in risks if r >= 0.8)} assignments")
            report_lines.append("")
        
        # High-risk assignments
        high_risk = [a for a in schedule.assignments if a.violation_risk_score >= 0.6]
        if high_risk:
            report_lines.append("HIGH-RISK ASSIGNMENTS (Risk >= 0.6)")
            report_lines.append("-" * 80)
            for assignment in sorted(high_risk, key=lambda x: x.violation_risk_score, reverse=True)[:10]:
                report_lines.append(
                    f"  Driver: {assignment.driver.name} | "
                    f"Route: {assignment.route.route_id} | "
                    f"Risk: {assignment.violation_risk_score:.3f} | "
                    f"Hours: {assignment.route.estimated_hours:.1f}"
                )
            report_lines.append("")
        
        # Unassigned routes
        if schedule.unassigned_routes:
            report_lines.append("UNASSIGNED ROUTES")
            report_lines.append("-" * 80)
            for route in schedule.unassigned_routes[:10]:
                report_lines.append(
                    f"  Route: {route.route_id} | "
                    f"Hours: {route.estimated_hours:.1f} | "
                    f"Priority: {route.priority}/10"
                )
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


# Main execution
if __name__ == "__main__":
    print("""
=================================================================================
FLEET OPTIMIZATION ENGINE - MODULE 12
=================================================================================

Intelligent scheduling and route optimization to minimize HOS violations.

FEATURES:
1. Genetic Algorithm Optimization
   - Population-based search
   - Multi-objective optimization
   - Handles complex constraints

2. Constraint Programming
   - Exact solutions for smaller problems
   - Guaranteed feasibility
   - Fast for assignment problems

3. Risk-Aware Scheduling
   - ML-predicted violation risk
   - Driver history consideration
   - HOS compliance constraints

4. Multiple Objectives
   - Minimize violations
   - Minimize costs
   - Maximize efficiency
   - Balance workload

USAGE:
from optimization_engine import AdvancedOptimizationEngine, Driver, Route

# Create drivers
drivers = [
    Driver('DRV_001', 'John Doe', experience_years=10, 
           violation_history=0, current_hours_worked_week=45),
    Driver('DRV_002', 'Jane Smith', experience_years=5,
           violation_history=1, current_hours_worked_week=50)
]

# Create routes
routes = [
    Route('RT_001', origin=(40.7, -74.0), destination=(34.0, -118.2),
          estimated_hours=8.5, priority=8, distance_miles=450),
    Route('RT_002', origin=(41.8, -87.6), destination=(29.7, -95.3),
          estimated_hours=10.0, priority=7, distance_miles=550)
]

# Optimize
engine = AdvancedOptimizationEngine(predictor=your_predictor)
schedule = engine.optimize_fleet_schedule(
    drivers, routes,
    objective=OptimizationObjective.MINIMIZE_VIOLATIONS,
    method='genetic'
)

# Generate report
report = engine.generate_optimization_report(schedule)
print(report)

# Get best driver for specific route
best_driver = engine.recommend_best_driver(routes[0], drivers)
print(f"Best driver for route: {best_driver.name}")

=================================================================================
    """)