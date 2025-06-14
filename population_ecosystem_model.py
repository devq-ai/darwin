#!/usr/bin/env python3
"""
Population Ecosystem Model with Stochastic Genetic Algorithm

This module implements a population ecosystem model that simulates customer growth,
app usage, and connection patterns. It uses a genetic algorithm with stochastic
parameters to optimize the model configuration.
"""

import asyncio
import datetime
import random

import panel as pn
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

# Initialize Panel extension if running in notebook
try:
    pn.extension()
except:
    pass

class PopulationEcosystemModel:
    """Population ecosystem model with stochastic genetic algorithm optimization"""

    def __init__(self, custom_params=None):
        """Initialize the model with default or custom parameters"""
        # Default parameters
        self.params = {
            # Population model parameters
            'initial_population': 1000,
            'growth_rate': 0.05,  # 5% weekly growth
            'carrying_capacity': 50000,
            'initial_apps_per_customer': 3,
            'app_addition_frequency': 4,  # New apps every 4 weeks
            'new_apps_per_period': [1, 5],  # Range of new apps per period
            'connection_capacity': 50000,  # Initial connection capacity
            'connection_capacity_range': [30000, 70000],  # Range for optimization
            'time_periods': 156,  # 3 years of weekly data

            # GA parameters
            'population_size': 50,
            'crossover_probability_range': [0.6, 0.8],  # Stochastic range
            'mutation_probability_range': [0.01, 0.1],  # Stochastic range
            'max_generations': 52,  # One year of weekly generations
            'early_stopping_generations': 10
        }

        # Update with custom parameters if provided
        if custom_params:
            self.params.update(custom_params)

        # Initialize simulation data
        self.simulation_data = {}
        self.optimization_results = None
        self.mcp_client = None

    async def connect(self):
        """Connect to Darwin MCP server"""
        try:
            # Import MCP client
            from darwin.mcp.client import DarwinMCPClient

            # Create client
            self.mcp_client = DarwinMCPClient()

            # Connect to server
            await self.mcp_client.connect("ws://localhost:8000/mcp")

            print("✅ Connected to Darwin MCP server")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Darwin MCP server: {e}")
            print("⚠️ Continuing with local simulation only")
            return False

    async def setup_optimization_problem(self):
        """Set up the optimization problem"""
        if not self.mcp_client:
            print("⚠️ No MCP client available, skipping optimization setup")
            return False

        # Define optimization problem
        optimization_problem = {
            "name": "Population Ecosystem Optimization",
            "objective": "maximize",
            "variables": [
                {
                    "name": "connection_capacity",
                    "type": "integer",
                    "min": self.params['connection_capacity_range'][0],
                    "max": self.params['connection_capacity_range'][1],
                    "step": 1000
                },
                {
                    "name": "growth_rate",
                    "type": "float",
                    "min": 0.01,
                    "max": 0.1,
                    "step": 0.005
                }
            ],
            "constraints": [
                {
                    "name": "missed_connections",
                    "type": "less_than",
                    "value": 5.0  # Maximum 5% missed connections
                },
                {
                    "name": "total_cost",
                    "type": "less_than",
                    "value": 1000000  # Maximum $1M total cost
                }
            ],
            "ga_params": {
                "population_size": self.params['population_size'],
                "max_generations": self.params['max_generations'],
                "early_stopping_generations": self.params['early_stopping_generations'],
                "stochastic_params": True,
                "crossover_probability_range": self.params['crossover_probability_range'],
                "mutation_probability_range": self.params['mutation_probability_range']
            }
        }

        # Send to MCP server
        response = await self.mcp_client.setup_optimization(optimization_problem)

        if response.get("status") == "success":
            print(f"✅ Optimization problem set up: {response.get('problem_id')}")
            self.problem_id = response.get('problem_id')
            return True
        else:
            print(f"❌ Failed to set up optimization problem: {response}")
            return False

    async def run_optimization(self):
        """Run the optimization"""
        if not self.mcp_client or not hasattr(self, 'problem_id'):
            print("⚠️ No MCP client or problem ID available, skipping optimization")
            # Simulate optimization results
            self.simulate_optimization_results()
            return False

        # Start optimization
        response = await self.mcp_client.start_optimization(self.problem_id)

        if response.get("status") == "success":
            print(f"✅ Optimization started: {response.get('run_id')}")
            self.run_id = response.get('run_id')

            # Wait for optimization to complete
            print("⏳ Waiting for optimization to complete...")
            status = "running"
            while status == "running":
                await asyncio.sleep(1)
                status_response = await self.mcp_client.get_optimization_status(self.run_id)
                status = status_response.get("status")
                if status_response.get("progress"):
                    print(f"Progress: {status_response.get('progress')}%")

            if status == "completed":
                print("✅ Optimization completed")
                return True
            else:
                print(f"❌ Optimization failed: {status}")
                return False
        else:
            print(f"❌ Failed to start optimization: {response}")
            # Simulate optimization results
            self.simulate_optimization_results()
            return False

    def simulate_optimization_results(self):
        """Simulate optimization results for local testing"""
        print("⚠️ Simulating optimization results")

        # Generate random best solution
        connection_capacity = random.randint(
            self.params['connection_capacity_range'][0],
            self.params['connection_capacity_range'][1]
        )
        growth_rate = round(random.uniform(0.01, 0.1), 3)

        # Generate random fitness history
        generations = self.params['max_generations']
        best_fitness_history = [random.uniform(0.5, 0.7)]
        for _ in range(1, generations):
            improvement = random.uniform(0, 0.01)
            best_fitness_history.append(min(0.99, best_fitness_history[-1] + improvement))

        # Generate random crossover and mutation probabilities
        crossover_probs_history = []
        mutation_probs_history = []
        for _ in range(generations):
            crossover_probs_history.append(random.uniform(
                self.params['crossover_probability_range'][0],
                self.params['crossover_probability_range'][1]
            ))
            mutation_probs_history.append(random.uniform(
                self.params['mutation_probability_range'][0],
                self.params['mutation_probability_range'][1]
            ))

        # Store results
        self.optimization_results = {
            "best_solution": {
                "connection_capacity": connection_capacity,
                "growth_rate": growth_rate
            },
            "best_fitness": best_fitness_history[-1],
            "best_fitness_history": best_fitness_history,
            "crossover_probs_history": crossover_probs_history,
            "mutation_probs_history": mutation_probs_history,
            "generations": generations
        }

        # Update params with optimized values
        self.params['connection_capacity'] = connection_capacity
        self.params['growth_rate'] = growth_rate

        print(f"✅ Simulated optimization complete")
        print(f"📊 Best solution: connection_capacity={connection_capacity}, growth_rate={growth_rate}")
        print(f"📈 Best fitness: {best_fitness_history[-1]:.4f}")

    async def get_results(self):
        """Get optimization results"""
        if not self.mcp_client or not hasattr(self, 'run_id'):
            print("⚠️ No MCP client or run ID available, using simulated results")
            if not self.optimization_results:
                self.simulate_optimization_results()
            return self.optimization_results

        # Get results from MCP server
        response = await self.mcp_client.get_optimization_results(self.run_id)

        if response.get("status") == "success":
            print("✅ Got optimization results")
            self.optimization_results = response.get("results")

            # Update params with optimized values
            self.params['connection_capacity'] = self.optimization_results["best_solution"]["connection_capacity"]
            self.params['growth_rate'] = self.optimization_results["best_solution"]["growth_rate"]

            return self.optimization_results
        else:
            print(f"❌ Failed to get optimization results: {response}")
            if not self.optimization_results:
                self.simulate_optimization_results()
            return self.optimization_results

    def calculate_operating_cost(self, connections, capacity):
        """Calculate operating cost based on connections and capacity"""
        # Base cost proportional to capacity
        base_cost = capacity * 0.1

        # Variable cost based on actual connections
        variable_cost = connections * 0.05

        # Penalty for approaching capacity
        utilization = connections / capacity if capacity > 0 else 1.0
        if utilization > 0.8:
            penalty = base_cost * (utilization - 0.8) * 5
        else:
            penalty = 0

        return base_cost + variable_cost + penalty

    def run_simulation(self):
        """Run the population ecosystem simulation"""
        params = self.params

        # Initialize simulation variables
        time_periods = params['time_periods']
        population = [params['initial_population']]
        apps_per_customer = [params['initial_apps_per_customer']]
        total_connections = [population[0] * apps_per_customer[0] * 10]  # Assume 10 connections per app
        missed_connections = [0]
        missed_connections_pct = [0]
        operating_costs = [self.calculate_operating_cost(total_connections[0], params['connection_capacity'])]

        # Generate dates starting from today
        start_date = datetime.datetime.now().date()
        dates = [start_date + datetime.timedelta(weeks=i) for i in range(time_periods)]

        # Run simulation for each time period
        for t in range(1, time_periods):
            # Calculate new population using logistic growth model
            new_population = population[t-1] + params['growth_rate'] * population[t-1] * (1 - population[t-1] / params['carrying_capacity'])
            population.append(new_population)

            # Calculate new apps per customer
            new_apps = apps_per_customer[t-1]
            if t % params['app_addition_frequency'] == 0:
                # Add new apps periodically
                new_apps += random.uniform(params['new_apps_per_period'][0], params['new_apps_per_period'][1])
            apps_per_customer.append(new_apps)

            # Calculate connections
            connections = population[t] * apps_per_customer[t] * 10  # Assume 10 connections per app

            # Calculate missed connections if over capacity
            if connections > params['connection_capacity']:
                missed = connections - params['connection_capacity']
                missed_pct = (missed / connections) * 100
            else:
                missed = 0
                missed_pct = 0

            total_connections.append(connections)
            missed_connections.append(missed)
            missed_connections_pct.append(missed_pct)

            # Calculate operating cost
            cost = self.calculate_operating_cost(connections, params['connection_capacity'])
            operating_costs.append(cost)

        # Store simulation data
        self.simulation_data = {
            'dates': dates,
            'population': population,
            'apps_per_customer': apps_per_customer,
            'total_connections': total_connections,
            'missed_connections': missed_connections,
            'missed_connections_pct': missed_connections_pct,
            'operating_costs': operating_costs
        }

        print(f"✅ Simulation complete: {time_periods} time periods")
        print(f"📊 Final population: {population[-1]:.0f}")
        print(f"📱 Final apps per customer: {apps_per_customer[-1]:.2f}")
        print(f"🔌 Final connections: {total_connections[-1]:.0f}")
        print(f"❌ Final missed connections: {missed_connections_pct[-1]:.2f}%")
        print(f"💰 Final operating cost: ${operating_costs[-1]:.2f}")

        return self.simulation_data

    def create_dashboard(self):
        """Create an interactive dashboard for the simulation results"""
        if not self.simulation_data:
            print("⚠️ No simulation data available, running simulation first")
            self.run_simulation()

        # Get simulation data
        data = self.simulation_data
        params = self.params

        # Create dashboard
        dashboard = pn.template.FastListTemplate(
            title="Population Ecosystem Model",
            sidebar_width=350
        )

        # Create parameter panel
        param_panel = pn.Column(
            pn.pane.Markdown("## Model Parameters"),
            pn.pane.Markdown(f"**Initial Population:** {params['initial_population']:,}"),
            pn.pane.Markdown(f"**Growth Rate:** {params['growth_rate']:.2%}"),
            pn.pane.Markdown(f"**Carrying Capacity:** {params['carrying_capacity']:,}"),
            pn.pane.Markdown(f"**Initial Apps/Customer:** {params['initial_apps_per_customer']}"),
            pn.pane.Markdown(f"**App Addition Frequency:** Every {params['app_addition_frequency']} weeks"),
            pn.pane.Markdown(f"**New Apps Range:** {params['new_apps_per_period'][0]}-{params['new_apps_per_period'][1]}"),
            pn.pane.Markdown(f"**Connection Capacity:** {params['connection_capacity']:,}"),
            pn.pane.Markdown("## GA Parameters"),
            pn.pane.Markdown(f"**Population Size:** {params['population_size']}"),
            pn.pane.Markdown(f"**Crossover Prob Range:** {params['crossover_probability_range'][0]}-{params['crossover_probability_range'][1]}"),
            pn.pane.Markdown(f"**Mutation Prob Range:** {params['mutation_probability_range'][0]}-{params['mutation_probability_range'][1]}"),
            pn.pane.Markdown(f"**Max Generations:** {params['max_generations']}"),
            width=300
        )

        # Extract data for plotting
        dates = data['dates']
        population = data['population']
        avg_apps = data['apps_per_customer']
        total_connections = data['total_connections']
        missed_connections_pct = data['missed_connections_pct']
        operating_costs = data['operating_costs']

        # Format dates for display
        date_strings = [d.strftime('%Y-%m-%d') for d in dates]
        time_periods = len(dates)

        # Create time slider
        time_slider = pn.widgets.IntSlider(
            name='Week',
            start=0,
            end=time_periods-1,
            value=0,
            width=800
        )

        # Create Bokeh figure for population growth
        pop_fig = figure(
            title="Customer Growth Over Time",
            x_axis_label="Date",
            y_axis_label="Customers",
            width=800,
            height=300
        )

        pop_source = ColumnDataSource(data={
            'date': date_strings,
            'population': population,
            'avg_apps': avg_apps,
            'connections': total_connections,
            'missed_pct': missed_connections_pct,
            'costs': operating_costs
        })

        pop_fig.line('date', 'population', source=pop_source, line_width=2, color='blue', legend_label='Customers')
        pop_fig.add_tools(HoverTool(
            tooltips=[
                ('Date', '@date'),
                ('Customers', '@population{0,0}'),
                ('Avg Apps/Customer', '@avg_apps{0.00}'),
                ('Total Connections', '@connections{0,0}')
            ]
        ))

        # Add carrying capacity line
        pop_fig.line(
            x=date_strings,
            y=[params['carrying_capacity']] * time_periods,
            line_width=1.5,
            color='red',
            line_dash='dashed',
            legend_label='Carrying Capacity'
        )

        pop_fig.legend.location = "bottom_right"
        pop_fig.xaxis.major_label_orientation = 45

        # Create Bokeh figure for missed connections
        missed_fig = figure(
            title="Missed Connections Percentage",
            x_axis_label="Date",
            y_axis_label="Missed Connections (%)",
            width=800,
            height=300
        )

        missed_fig.line('date', 'missed_pct', source=pop_source, line_width=2, color='red')
        missed_fig.y_range.start = 0
        missed_fig.y_range.end = max(100, max(missed_connections_pct) * 1.1)
        missed_fig.add_tools(HoverTool(
            tooltips=[
                ('Date', '@date'),
                ('Missed Connections', '@missed_pct{0.00}%')
            ]
        ))
        missed_fig.xaxis.major_label_orientation = 45

        # Create Bokeh figure for operating costs
        cost_fig = figure(
            title="Weekly Operating Costs",
            x_axis_label="Date",
            y_axis_label="Cost ($)",
            width=800,
            height=300
        )

        cost_fig.line('date', 'costs', source=pop_source, line_width=2, color='green')
        cost_fig.add_tools(HoverTool(
            tooltips=[
                ('Date', '@date'),
                ('Operating Cost', '$@costs{0,0.00}')
            ]
        ))
        cost_fig.xaxis.major_label_orientation = 45

        # Create Bokeh figure for connections
        conn_fig = figure(
            title="Total Connections vs Capacity",
            x_axis_label="Date",
            y_axis_label="Connections",
            width=800,
            height=300
        )

        conn_fig.line('date', 'connections', source=pop_source, line_width=2, color='purple', legend_label='Total Connections')

        # Add connection capacity line
        conn_fig.line(
            'date', 'capacity',
            source=pop_source,
            line_width=2,
            color='red',
            legend_label='Connection Capacity',
            line_dash='dashed'
        )

        # Add all components to dashboard
        dashboard.sidebar.append(param_panel)
        dashboard.main.append(
            pn.Column(
                pn.Row(time_slider),
                pn.pane.Bokeh(pop_fig),
                pn.pane.Bokeh(apps_fig),
                pn.pane.Bokeh(cost_fig),
                pn.pane.Bokeh(conn_fig)
            )
        )

        return dashboard
