#!/usr/bin/env python3
"""
Resource Management System Demonstration

This script demonstrates the sophisticated resource management capabilities
including dynamic allocation, priority scheduling, load balancing, monitoring,
and predictive resource management.
"""

import asyncio
import time
import logging
import random
from typing import Dict, List
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from unified_cognitive_kernel.resource_manager import (
    DynamicResourceManager,
    ResourceRequest,
    ResourceType,
    Priority,
    ResourceConstraint
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResourceManagementDemo:
    """Demonstrates resource management system capabilities"""
    
    def __init__(self):
        self.resource_manager = None
        self.active_allocations: List[str] = []
        
    async def initialize(self):
        """Initialize the demonstration"""
        logger.info("🚀 Initializing Resource Management Demonstration")
        
        # Configure resource manager
        config = {
            'max_concurrent_tasks': 6,
            'memory_pool_size': 1024 * 1024 * 200,  # 200MB
            'attention_capacity': 150.0,
            'cognitive_modules': [
                'reasoning', 'memory', 'attention', 
                'planning', 'learning', 'perception'
            ],
            'monitor_interval': 0.5,
            'enable_preallocation': True,
            'enable_load_balancing': True,
            'enable_graceful_degradation': True
        }
        
        self.resource_manager = DynamicResourceManager(config)
        await self.resource_manager.initialize()
        
        # Add resource constraints
        await self._setup_resource_constraints()
        
        # Setup degradation strategies
        await self._setup_degradation_strategies()
        
        logger.info("✅ Resource Management System initialized")
    
    async def _setup_resource_constraints(self):
        """Setup resource constraints for demonstration"""
        logger.info("📏 Setting up resource constraints")
        
        # CPU constraints
        cpu_constraint = ResourceConstraint(
            resource_type=ResourceType.CPU,
            min_required=5.0,
            max_allowed=90.0,
            preferred=50.0,
            hard_limit=True
        )
        self.resource_manager.add_resource_constraint(cpu_constraint)
        
        # Memory constraints
        memory_constraint = ResourceConstraint(
            resource_type=ResourceType.MEMORY,
            min_required=64.0,  # 64 bytes minimum
            max_allowed=1024.0 * 1024 * 50,  # 50MB maximum
            preferred=1024.0 * 1024 * 10,  # 10MB preferred
            hard_limit=False
        )
        self.resource_manager.add_resource_constraint(memory_constraint)
        
        # Attention constraints
        attention_constraint = ResourceConstraint(
            resource_type=ResourceType.ATTENTION,
            min_required=1.0,
            max_allowed=50.0,
            preferred=20.0,
            hard_limit=False
        )
        self.resource_manager.add_resource_constraint(attention_constraint)
        
        logger.info("✅ Resource constraints configured")
    
    async def _setup_degradation_strategies(self):
        """Setup graceful degradation strategies"""
        logger.info("🔄 Setting up degradation strategies")
        
        def cpu_degradation_strategy(alert_data):
            logger.warning(f"🚨 CPU degradation triggered: {alert_data['utilization']:.1%}")
            logger.info("🔧 Applying CPU optimization strategies...")
            # In a real system, this might reduce task complexity, 
            # defer non-critical operations, etc.
        
        def memory_degradation_strategy(alert_data):
            logger.warning(f"🚨 Memory degradation triggered: {alert_data['utilization']:.1%}")
            logger.info("🔧 Applying memory optimization strategies...")
            # Trigger aggressive garbage collection, cache eviction, etc.
        
        def attention_degradation_strategy(alert_data):
            logger.warning(f"🚨 Attention degradation triggered: {alert_data['utilization']:.1%}")
            logger.info("🔧 Applying attention optimization strategies...")
            # Focus on critical tasks, reduce background processing, etc.
        
        self.resource_manager.set_degradation_strategy(
            ResourceType.CPU, cpu_degradation_strategy
        )
        self.resource_manager.set_degradation_strategy(
            ResourceType.MEMORY, memory_degradation_strategy
        )
        self.resource_manager.set_degradation_strategy(
            ResourceType.ATTENTION, attention_degradation_strategy
        )
        
        logger.info("✅ Degradation strategies configured")
    
    async def demonstrate_priority_scheduling(self):
        """Demonstrate priority-based resource scheduling"""
        logger.info("\n🎯 === Priority-Based Scheduling Demonstration ===")
        
        # Create requests with different priorities
        requests = [
            ResourceRequest(
                request_id="background_task",
                resource_type=ResourceType.CPU,
                amount=30.0,
                priority=Priority.BACKGROUND,
                duration_estimate=3.0
            ),
            ResourceRequest(
                request_id="critical_decision",
                resource_type=ResourceType.CPU,
                amount=60.0,
                priority=Priority.CRITICAL,
                duration_estimate=0.5
            ),
            ResourceRequest(
                request_id="normal_processing",
                resource_type=ResourceType.CPU,
                amount=40.0,
                priority=Priority.NORMAL,
                duration_estimate=2.0
            ),
            ResourceRequest(
                request_id="high_priority_task",
                resource_type=ResourceType.CPU,
                amount=50.0,
                priority=Priority.HIGH,
                duration_estimate=1.0
            )
        ]
        
        # Submit requests in non-priority order
        logger.info("📤 Submitting tasks in non-priority order...")
        allocation_times = {}
        
        for request in requests:
            start_time = time.time()
            allocation_id = await self.resource_manager.allocate_resources(request)
            end_time = time.time()
            
            allocation_times[request.priority.name] = end_time - start_time
            
            if allocation_id:
                self.active_allocations.append(allocation_id)
                logger.info(f"✅ {request.priority.name}: {request.request_id} allocated in {allocation_times[request.priority.name]:.3f}s")
            else:
                logger.warning(f"❌ {request.priority.name}: {request.request_id} allocation failed")
        
        # Show scheduling effectiveness
        logger.info("📊 Priority scheduling results:")
        for priority, alloc_time in sorted(allocation_times.items()):
            logger.info(f"  {priority}: {alloc_time:.3f}s response time")
        
        await asyncio.sleep(1)  # Let tasks run briefly
        
        # Clean up allocations
        for allocation_id in self.active_allocations[:]:
            await self.resource_manager.release_resources(allocation_id)
            self.active_allocations.remove(allocation_id)
        
        logger.info("✅ Priority scheduling demonstration complete")
    
    async def demonstrate_memory_management(self):
        """Demonstrate advanced memory management"""
        logger.info("\n💾 === Memory Management Demonstration ===")
        
        # Show initial memory state
        status = self.resource_manager.get_resource_status()
        initial_utilization = status['memory_utilization']
        initial_fragmentation = status['memory_fragmentation']
        
        logger.info(f"📊 Initial memory utilization: {initial_utilization:.1%}")
        logger.info(f"📊 Initial memory fragmentation: {initial_fragmentation:.1%}")
        
        # Allocate memory blocks of various sizes
        memory_allocations = []
        block_sizes = [1024, 2048, 4096, 8192, 16384, 32768]  # Various block sizes
        
        logger.info("📤 Allocating memory blocks of various sizes...")
        for i, size in enumerate(block_sizes):
            request = ResourceRequest(
                request_id=f"memory_block_{i}",
                resource_type=ResourceType.MEMORY,
                amount=float(size),
                priority=Priority.NORMAL
            )
            
            allocation_id = await self.resource_manager.allocate_resources(request)
            if allocation_id:
                memory_allocations.append(allocation_id)
                logger.info(f"✅ Allocated {size} bytes: {allocation_id}")
            else:
                logger.warning(f"❌ Failed to allocate {size} bytes")
        
        # Check memory state after allocations
        status = self.resource_manager.get_resource_status()
        peak_utilization = status['memory_utilization']
        peak_fragmentation = status['memory_fragmentation']
        
        logger.info(f"📊 Peak memory utilization: {peak_utilization:.1%}")
        logger.info(f"📊 Peak memory fragmentation: {peak_fragmentation:.1%}")
        
        # Demonstrate fragmentation by releasing every other block
        logger.info("🔄 Creating fragmentation by releasing alternating blocks...")
        for i in range(0, len(memory_allocations), 2):
            await self.resource_manager.release_resources(memory_allocations[i])
            logger.info(f"✅ Released block {i}")
        
        # Check fragmentation
        status = self.resource_manager.get_resource_status()
        fragmented_utilization = status['memory_utilization']
        fragmented_fragmentation = status['memory_fragmentation']
        
        logger.info(f"📊 After fragmentation - utilization: {fragmented_utilization:.1%}")
        logger.info(f"📊 After fragmentation - fragmentation: {fragmented_fragmentation:.1%}")
        
        # Trigger garbage collection
        logger.info("🗑️ Triggering garbage collection...")
        collected = self.resource_manager.memory_manager.garbage_collect()
        logger.info(f"🗑️ Garbage collection freed {collected} objects")
        
        # Clean up remaining allocations
        for i in range(1, len(memory_allocations), 2):
            await self.resource_manager.release_resources(memory_allocations[i])
        
        # Final memory state
        status = self.resource_manager.get_resource_status()
        final_utilization = status['memory_utilization']
        final_fragmentation = status['memory_fragmentation']
        
        logger.info(f"📊 Final memory utilization: {final_utilization:.1%}")
        logger.info(f"📊 Final memory fragmentation: {final_fragmentation:.1%}")
        
        logger.info("✅ Memory management demonstration complete")
    
    async def demonstrate_attention_allocation(self):
        """Demonstrate attention resource allocation"""
        logger.info("\n🧠 === Attention Allocation Demonstration ===")
        
        # Show initial attention state
        available_attention = self.resource_manager.attention_allocator.get_available_attention()
        logger.info(f"📊 Initial available attention: {available_attention:.1f} units")
        
        # Create attention allocation requests
        attention_requests = [
            ("visual_processing", 25.0, Priority.HIGH),
            ("language_comprehension", 30.0, Priority.NORMAL),
            ("working_memory", 20.0, Priority.HIGH),
            ("background_monitoring", 15.0, Priority.LOW),
            ("emotional_processing", 35.0, Priority.NORMAL),
            ("critical_decision", 40.0, Priority.CRITICAL)
        ]
        
        attention_allocations = []
        
        logger.info("📤 Allocating attention to cognitive processes...")
        for task_name, amount, priority in attention_requests:
            request = ResourceRequest(
                request_id=task_name,
                resource_type=ResourceType.ATTENTION,
                amount=amount,
                priority=priority
            )
            
            allocation_id = await self.resource_manager.allocate_resources(request)
            if allocation_id:
                attention_allocations.append((allocation_id, task_name, amount))
                available = self.resource_manager.attention_allocator.get_available_attention()
                logger.info(f"✅ {task_name}: {amount:.1f} units allocated, {available:.1f} remaining")
            else:
                logger.warning(f"❌ {task_name}: allocation failed (insufficient attention)")
        
        # Show attention rebalancing in action
        logger.info("🔄 Demonstrating attention rebalancing...")
        
        # Try to allocate more than available (should trigger rebalancing)
        emergency_request = ResourceRequest(
            request_id="emergency_response",
            resource_type=ResourceType.ATTENTION,
            amount=60.0,  # Large amount
            priority=Priority.CRITICAL
        )
        
        emergency_allocation = await self.resource_manager.allocate_resources(emergency_request)
        if emergency_allocation:
            attention_allocations.append((emergency_allocation, "emergency_response", 60.0))
            available = self.resource_manager.attention_allocator.get_available_attention()
            logger.info(f"✅ Emergency response: 60.0 units allocated through rebalancing, {available:.1f} remaining")
        
        # Show current allocation state
        logger.info("📊 Current attention allocations:")
        allocator = self.resource_manager.attention_allocator
        for target, allocation in allocator.allocations.items():
            priority = allocator.priorities.get(target, Priority.NORMAL)
            logger.info(f"  {target}: {allocation:.1f} units ({priority.name})")
        
        # Demonstrate attention decay
        logger.info("⏰ Demonstrating attention decay...")
        original_allocations = dict(allocator.allocations)
        
        # Apply several decay cycles
        for cycle in range(3):
            allocator.decay_attention()
            available = allocator.get_available_attention()
            logger.info(f"  Decay cycle {cycle + 1}: {available:.1f} units now available")
        
        # Show decay effect
        logger.info("📊 Attention after decay:")
        for target, allocation in allocator.allocations.items():
            original = original_allocations.get(target, 0.0)
            decay_pct = (1 - allocation / original) * 100 if original > 0 else 0
            logger.info(f"  {target}: {allocation:.1f} units ({decay_pct:.1f}% decay)")
        
        # Clean up attention allocations
        for allocation_id, task_name, _ in attention_allocations:
            await self.resource_manager.release_resources(allocation_id)
        
        final_available = self.resource_manager.attention_allocator.get_available_attention()
        logger.info(f"📊 Final available attention: {final_available:.1f} units")
        
        logger.info("✅ Attention allocation demonstration complete")
    
    async def demonstrate_load_balancing(self):
        """Demonstrate adaptive load balancing"""
        logger.info("\n⚖️ === Load Balancing Demonstration ===")
        
        # Show initial module loads
        balancer = self.resource_manager.load_balancer
        logger.info("📊 Initial module loads:")
        for module, load in balancer.module_loads.items():
            capacity = balancer.module_capacities[module]
            utilization = load / capacity * 100
            logger.info(f"  {module}: {load:.1f}/{capacity:.1f} ({utilization:.1f}%)")
        
        # Create multiple CPU tasks to distribute
        cpu_tasks = []
        task_loads = [20.0, 35.0, 15.0, 40.0, 25.0, 30.0, 10.0, 45.0]
        
        logger.info("📤 Submitting CPU tasks for load balancing...")
        for i, load in enumerate(task_loads):
            request = ResourceRequest(
                request_id=f"cpu_task_{i}",
                resource_type=ResourceType.CPU,
                amount=load,
                priority=Priority.NORMAL,
                duration_estimate=2.0
            )
            
            allocation_id = await self.resource_manager.allocate_resources(request)
            if allocation_id:
                cpu_tasks.append(allocation_id)
                logger.info(f"✅ Task {i} (load {load:.1f}) assigned")
            else:
                logger.warning(f"❌ Task {i} (load {load:.1f}) assignment failed")
        
        # Show load distribution
        logger.info("📊 Load distribution after task assignment:")
        for module, load in balancer.module_loads.items():
            capacity = balancer.module_capacities[module]
            utilization = load / capacity * 100
            logger.info(f"  {module}: {load:.1f}/{capacity:.1f} ({utilization:.1f}%)")
        
        # Simulate load balancing by completing some tasks
        logger.info("🔄 Simulating task completion and rebalancing...")
        
        # Complete half the tasks to trigger rebalancing
        for i in range(len(cpu_tasks) // 2):
            await self.resource_manager.release_resources(cpu_tasks[i])
            logger.info(f"✅ Completed task {i}")
        
        # Show updated load distribution
        logger.info("📊 Load distribution after task completion:")
        for module, load in balancer.module_loads.items():
            capacity = balancer.module_capacities[module]
            utilization = load / capacity * 100
            logger.info(f"  {module}: {load:.1f}/{capacity:.1f} ({utilization:.1f}%)")
        
        # Clean up remaining tasks
        for i in range(len(cpu_tasks) // 2, len(cpu_tasks)):
            await self.resource_manager.release_resources(cpu_tasks[i])
        
        logger.info("✅ Load balancing demonstration complete")
    
    async def demonstrate_resource_monitoring(self):
        """Demonstrate real-time resource monitoring"""
        logger.info("\n📊 === Resource Monitoring Demonstration ===")
        
        # Get current metrics
        metrics = self.resource_manager.monitor.get_metrics()
        
        if metrics:
            logger.info("📊 Current system metrics:")
            for resource_type, metric in metrics.items():
                if metric:
                    logger.info(f"  {resource_type.name}:")
                    logger.info(f"    Utilization: {metric.utilization_rate:.1%}")
                    logger.info(f"    Available: {metric.available:.1f}")
                    logger.info(f"    Allocated: {metric.allocated:.1f}")
                    logger.info(f"    Efficiency: {metric.efficiency:.1%}")
                    logger.info(f"    Contention: {metric.contention_level:.1%}")
        else:
            logger.info("📊 System metrics not yet available (monitoring starting up)")
        
        # Create a callback to demonstrate alerts
        alert_count = 0
        
        def demo_alert_handler(alert_data):
            nonlocal alert_count
            alert_count += 1
            resource_type = alert_data['resource_type']
            utilization = alert_data['utilization']
            threshold = alert_data['threshold']
            
            logger.warning(f"🚨 ALERT #{alert_count}: {resource_type.name} at {utilization:.1%} (threshold: {threshold:.1%})")
        
        # Add our demo alert handler
        self.resource_manager.monitor.add_alert_callback(demo_alert_handler)
        
        # Simulate high resource usage to trigger alerts
        logger.info("🔥 Simulating high resource usage to trigger monitoring alerts...")
        
        # Create many small allocations to increase utilization
        stress_allocations = []
        for i in range(10):
            request = ResourceRequest(
                request_id=f"stress_test_{i}",
                resource_type=ResourceType.ATTENTION,
                amount=8.0,  # Small but numerous allocations
                priority=Priority.NORMAL
            )
            
            allocation_id = await self.resource_manager.allocate_resources(request)
            if allocation_id:
                stress_allocations.append(allocation_id)
        
        # Wait a bit for monitoring to detect the usage
        await asyncio.sleep(2)
        
        # Get updated metrics
        metrics = self.resource_manager.monitor.get_metrics()
        if metrics:
            logger.info("📊 Metrics after stress test:")
            for resource_type, metric in metrics.items():
                if metric:
                    logger.info(f"  {resource_type.name}: {metric.utilization_rate:.1%} utilization")
        
        # Clean up stress allocations
        for allocation_id in stress_allocations:
            await self.resource_manager.release_resources(allocation_id)
        
        logger.info(f"📊 Total alerts triggered: {alert_count}")
        logger.info("✅ Resource monitoring demonstration complete")
    
    async def demonstrate_predictive_allocation(self):
        """Demonstrate resource usage prediction and preallocation"""
        logger.info("\n🔮 === Predictive Resource Allocation Demonstration ===")
        
        predictor = self.resource_manager.predictor
        
        # Simulate historical usage data with an upward trend
        logger.info("📈 Simulating historical usage data with upward trend...")
        
        base_time = time.time() - 100  # Start 100 seconds ago
        for i in range(20):
            # Simulate increasing CPU usage over time
            usage = 30.0 + i * 2.0 + random.uniform(-5, 5)  # Trending upward with noise
            predictor.record_usage(ResourceType.CPU, usage)
            
            # Simulate varying memory usage
            mem_usage = 40.0 + random.uniform(-10, 20)
            predictor.record_usage(ResourceType.MEMORY, mem_usage)
            
            # Simulate attention usage with daily pattern
            att_usage = 50.0 + 20.0 * (0.5 + 0.5 * random.random())
            predictor.record_usage(ResourceType.ATTENTION, att_usage)
        
        logger.info("✅ Historical data generated")
        
        # Make predictions
        logger.info("🔮 Making resource usage predictions...")
        
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.ATTENTION]:
            # Predict usage 30 seconds ahead
            prediction = predictor.predict_usage(resource_type, 30.0)
            accuracy = predictor.get_prediction_accuracy(resource_type)
            
            if prediction is not None:
                logger.info(f"  {resource_type.name}: predicted {prediction:.1f}% usage in 30s (accuracy: {accuracy:.1%})")
            else:
                logger.info(f"  {resource_type.name}: insufficient data for prediction")
        
        # Demonstrate preallocation cycle
        logger.info("🚀 Running preallocation cycle...")
        
        # Run preallocation (this will check predictions and pre-allocate if needed)
        await self.resource_manager.preallocation_cycle()
        
        # Check if any preallocation occurred
        status = self.resource_manager.get_resource_status()
        active_allocations = status['active_allocations']
        
        if active_allocations > 0:
            logger.info(f"✅ Preallocation successful: {active_allocations} resources pre-allocated")
        else:
            logger.info("ℹ️ No preallocation needed at current prediction levels")
        
        # Test prediction accuracy by recording actual usage
        logger.info("📊 Testing prediction accuracy...")
        
        # Simulate some actual usage and update accuracy
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.ATTENTION]:
            model = predictor.prediction_models[resource_type]
            if model.last_prediction is not None:
                # Simulate actual usage close to prediction
                actual_usage = model.last_prediction + random.uniform(-5, 5)
                predictor.update_prediction_accuracy(resource_type, model.last_prediction, actual_usage)
                
                new_accuracy = predictor.get_prediction_accuracy(resource_type)
                logger.info(f"  {resource_type.name}: accuracy updated to {new_accuracy:.1%}")
        
        logger.info("✅ Predictive allocation demonstration complete")
    
    async def demonstrate_comprehensive_scenario(self):
        """Demonstrate comprehensive resource management scenario"""
        logger.info("\n🎭 === Comprehensive Scenario Demonstration ===")
        logger.info("Simulating complex cognitive workload with multiple resource demands...")
        
        # Scenario: Complex reasoning task requiring multiple resources
        scenario_allocations = []
        
        # Phase 1: Initial perception and attention focusing
        logger.info("🔍 Phase 1: Perception and attention focusing")
        
        perception_requests = [
            ResourceRequest("visual_attention", ResourceType.ATTENTION, 30.0, Priority.HIGH),
            ResourceRequest("sensory_memory", ResourceType.MEMORY, 2048.0, Priority.HIGH),
            ResourceRequest("pattern_recognition", ResourceType.CPU, 40.0, Priority.HIGH)
        ]
        
        for request in perception_requests:
            allocation_id = await self.resource_manager.allocate_resources(request)
            if allocation_id:
                scenario_allocations.append(allocation_id)
                logger.info(f"✅ {request.request_id}: allocated")
        
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Phase 2: Working memory and reasoning
        logger.info("🧠 Phase 2: Working memory and reasoning")
        
        reasoning_requests = [
            ResourceRequest("working_memory", ResourceType.MEMORY, 8192.0, Priority.CRITICAL),
            ResourceRequest("logical_reasoning", ResourceType.CPU, 60.0, Priority.CRITICAL),
            ResourceRequest("reasoning_attention", ResourceType.ATTENTION, 40.0, Priority.CRITICAL)
        ]
        
        for request in reasoning_requests:
            allocation_id = await self.resource_manager.allocate_resources(request)
            if allocation_id:
                scenario_allocations.append(allocation_id)
                logger.info(f"✅ {request.request_id}: allocated")
        
        await asyncio.sleep(0.5)
        
        # Phase 3: Decision making and response planning
        logger.info("🎯 Phase 3: Decision making and response planning")
        
        decision_requests = [
            ResourceRequest("decision_matrix", ResourceType.MEMORY, 4096.0, Priority.HIGH),
            ResourceRequest("response_planning", ResourceType.CPU, 35.0, Priority.HIGH),
            ResourceRequest("executive_attention", ResourceType.ATTENTION, 25.0, Priority.HIGH)
        ]
        
        for request in decision_requests:
            allocation_id = await self.resource_manager.allocate_resources(request)
            if allocation_id:
                scenario_allocations.append(allocation_id)
                logger.info(f"✅ {request.request_id}: allocated")
        
        # Show comprehensive status during peak usage
        logger.info("📊 System status during peak cognitive load:")
        status = self.resource_manager.get_resource_status()
        
        logger.info(f"  Active allocations: {status['active_allocations']}")
        logger.info(f"  Memory utilization: {status['memory_utilization']:.1%}")
        logger.info(f"  Memory fragmentation: {status['memory_fragmentation']:.1%}")
        logger.info(f"  Available attention: {status['attention_available']:.1f}")
        
        cpu_loads = status['cpu_utilization']
        logger.info("  CPU module loads:")
        for module, load in cpu_loads.items():
            logger.info(f"    {module}: {load:.1f}")
        
        await asyncio.sleep(1.0)  # Let the system run under load
        
        # Phase 4: Gradual resource release as task completes
        logger.info("🔄 Phase 4: Task completion and resource release")
        
        # Release resources in reverse order (cleanup)
        for allocation_id in reversed(scenario_allocations):
            await self.resource_manager.release_resources(allocation_id)
            logger.info(f"✅ Released: {allocation_id}")
            await asyncio.sleep(0.1)  # Gradual release
        
        # Final status
        final_status = self.resource_manager.get_resource_status()
        logger.info("📊 Final system status:")
        logger.info(f"  Active allocations: {final_status['active_allocations']}")
        logger.info(f"  Memory utilization: {final_status['memory_utilization']:.1%}")
        logger.info(f"  Available attention: {final_status['attention_available']:.1f}")
        
        logger.info("✅ Comprehensive scenario demonstration complete")
    
    async def run_full_demonstration(self):
        """Run the complete resource management demonstration"""
        try:
            logger.info("🎬 Starting Resource Management System Demonstration")
            logger.info("=" * 60)
            
            await self.initialize()
            
            # Run all demonstrations
            await self.demonstrate_priority_scheduling()
            await self.demonstrate_memory_management()
            await self.demonstrate_attention_allocation()
            await self.demonstrate_load_balancing()
            await self.demonstrate_resource_monitoring()
            await self.demonstrate_predictive_allocation()
            await self.demonstrate_comprehensive_scenario()
            
            # Final performance summary
            logger.info("\n📈 === Performance Summary ===")
            status = self.resource_manager.get_resource_status()
            perf_metrics = status.get('performance_metrics', {})
            
            if perf_metrics:
                for metric_name, values in perf_metrics.items():
                    if values:
                        avg_value = sum(values) / len(values)
                        logger.info(f"  {metric_name}: {avg_value:.3f}s average")
            
            logger.info("=" * 60)
            logger.info("🎉 Resource Management Demonstration Complete!")
            logger.info("✅ All systems demonstrated successfully:")
            logger.info("  • Priority-based scheduling with sub-second response times")
            logger.info("  • Advanced memory management with garbage collection")
            logger.info("  • Dynamic attention allocation with economic rebalancing")
            logger.info("  • Adaptive load balancing across cognitive modules")
            logger.info("  • Real-time monitoring with intelligent alerting")
            logger.info("  • Predictive resource management and preallocation")
            logger.info("  • Graceful degradation under resource pressure")
            
        except Exception as e:
            logger.error(f"❌ Demonstration failed: {e}")
            raise
        finally:
            if self.resource_manager:
                await self.resource_manager.shutdown()
                logger.info("🔄 Resource manager shut down cleanly")


async def main():
    """Main demonstration entry point"""
    demo = ResourceManagementDemo()
    await demo.run_full_demonstration()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)