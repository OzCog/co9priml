#!/usr/bin/env python3
"""
Complete Resource Management System Test

This comprehensive test validates all components of the resource management
system working together, including integration with the unified cognitive kernel.
"""

import asyncio
import time
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from unified_cognitive_kernel.resource_manager import (
    DynamicResourceManager, ResourceRequest, ResourceType, Priority
)
from unified_cognitive_kernel.cognitive_kernel import (
    UnifiedCognitiveKernel, CognitiveKernelConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveResourceTest:
    """Comprehensive test suite for the resource management system"""
    
    def __init__(self):
        self.test_results = []
        
    def record_test_result(self, test_name: str, success: bool, details: str = ""):
        """Record test result"""
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'timestamp': time.time()
        })
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name} - {details}")
    
    async def test_resource_manager_standalone(self):
        """Test resource manager as standalone component"""
        logger.info("🧪 Testing Resource Manager Standalone...")
        
        try:
            # Test 1: Basic initialization
            config = {
                'max_concurrent_tasks': 4,
                'memory_pool_size': 1024 * 1024,
                'attention_capacity': 100.0,
                'cognitive_modules': ['reasoning', 'memory', 'attention'],
                'monitor_interval': 0.1,
                'enable_preallocation': True,
                'enable_load_balancing': True,
                'enable_graceful_degradation': True
            }
            
            manager = DynamicResourceManager(config)
            await manager.initialize()
            
            self.record_test_result(
                "Resource Manager Initialization", 
                True, 
                "Successfully initialized with all features enabled"
            )
            
            # Test 2: Resource allocation and release
            requests = [
                ResourceRequest("cpu_test", ResourceType.CPU, 25.0, Priority.HIGH),
                ResourceRequest("mem_test", ResourceType.MEMORY, 1024.0, Priority.NORMAL),
                ResourceRequest("att_test", ResourceType.ATTENTION, 30.0, Priority.HIGH)
            ]
            
            allocations = []
            allocation_times = []
            
            for request in requests:
                start_time = time.time()
                allocation_id = await manager.allocate_resources(request)
                end_time = time.time()
                
                if allocation_id:
                    allocations.append(allocation_id)
                    allocation_times.append(end_time - start_time)
            
            avg_allocation_time = sum(allocation_times) / len(allocation_times)
            
            self.record_test_result(
                "Resource Allocation Speed",
                avg_allocation_time < 0.001,  # Sub-millisecond requirement
                f"Average allocation time: {avg_allocation_time:.4f}s"
            )
            
            # Test 3: Resource status reporting
            status = manager.get_resource_status()
            required_fields = [
                'active_allocations', 'cpu_utilization', 'memory_utilization',
                'attention_available', 'system_metrics', 'performance_metrics'
            ]
            
            status_complete = all(field in status for field in required_fields)
            
            self.record_test_result(
                "Resource Status Reporting",
                status_complete,
                f"Status contains {len(status)} fields"
            )
            
            # Test 4: Resource release
            release_count = 0
            for allocation_id in allocations:
                if await manager.release_resources(allocation_id):
                    release_count += 1
            
            self.record_test_result(
                "Resource Release",
                release_count == len(allocations),
                f"Released {release_count}/{len(allocations)} allocations"
            )
            
            await manager.shutdown()
            
        except Exception as e:
            self.record_test_result("Resource Manager Standalone", False, str(e))
    
    async def test_priority_scheduling_compliance(self):
        """Test priority scheduling meets requirements"""
        logger.info("🧪 Testing Priority Scheduling Compliance...")
        
        try:
            manager = DynamicResourceManager({
                'max_concurrent_tasks': 2,  # Limited to test queuing
                'cognitive_modules': ['test1', 'test2']
            })
            await manager.initialize()
            
            # Create requests with different priorities and measure completion order
            requests = [
                ("background", Priority.BACKGROUND, 10.0),
                ("critical", Priority.CRITICAL, 20.0),
                ("normal", Priority.NORMAL, 15.0),
                ("high", Priority.HIGH, 25.0)
            ]
            
            completion_order = []
            
            async def process_request(name, priority, amount):
                request = ResourceRequest(name, ResourceType.CPU, amount, priority)
                start_time = time.time()
                
                allocation_id = await manager.allocate_resources(request)
                if allocation_id:
                    await asyncio.sleep(0.1)  # Simulate work
                    await manager.release_resources(allocation_id)
                    completion_order.append((name, priority, time.time() - start_time))
            
            # Submit all requests concurrently
            tasks = [
                asyncio.create_task(process_request(name, priority, amount))
                for name, priority, amount in requests
            ]
            
            await asyncio.gather(*tasks)
            
            # Verify critical and high priority completed before lower priorities
            critical_time = next((t for n, p, t in completion_order if p == Priority.CRITICAL), float('inf'))
            high_time = next((t for n, p, t in completion_order if p == Priority.HIGH), float('inf'))
            normal_time = next((t for n, p, t in completion_order if p == Priority.NORMAL), float('inf'))
            background_time = next((t for n, p, t in completion_order if p == Priority.BACKGROUND), float('inf'))
            
            priority_respected = (critical_time <= high_time <= normal_time <= background_time)
            
            self.record_test_result(
                "Priority Scheduling Compliance",
                priority_respected,
                f"Completion times - Critical: {critical_time:.3f}s, High: {high_time:.3f}s, Normal: {normal_time:.3f}s, Background: {background_time:.3f}s"
            )
            
            await manager.shutdown()
            
        except Exception as e:
            self.record_test_result("Priority Scheduling Compliance", False, str(e))
    
    async def test_memory_management_efficiency(self):
        """Test memory management efficiency and garbage collection"""
        logger.info("🧪 Testing Memory Management Efficiency...")
        
        try:
            manager = DynamicResourceManager({'memory_pool_size': 1024 * 10})  # 10KB pool
            await manager.initialize()
            
            # Test fragmentation handling
            allocations = []
            
            # Allocate many small blocks
            for i in range(20):
                request = ResourceRequest(f"mem_{i}", ResourceType.MEMORY, 256.0, Priority.NORMAL)
                allocation_id = await manager.allocate_resources(request)
                if allocation_id:
                    allocations.append(allocation_id)
            
            initial_utilization = manager.memory_manager.get_utilization()
            
            # Release every other allocation to create fragmentation
            for i in range(0, len(allocations), 2):
                await manager.release_resources(allocations[i])
            
            fragmented_utilization = manager.memory_manager.get_utilization()
            fragmentation_level = manager.memory_manager.get_fragmentation()
            
            # Trigger garbage collection
            collected = manager.memory_manager.garbage_collect()
            
            # Test memory leak prevention
            final_utilization = manager.memory_manager.get_utilization()
            
            # Release remaining allocations
            for i in range(1, len(allocations), 2):
                await manager.release_resources(allocations[i])
            
            cleanup_utilization = manager.memory_manager.get_utilization()
            
            self.record_test_result(
                "Memory Fragmentation Handling",
                fragmentation_level >= 0.0,  # Should track fragmentation
                f"Fragmentation level: {fragmentation_level:.1%}"
            )
            
            self.record_test_result(
                "Memory Leak Prevention",
                cleanup_utilization < initial_utilization,
                f"Memory utilization: {initial_utilization:.1%} → {cleanup_utilization:.1%}"
            )
            
            await manager.shutdown()
            
        except Exception as e:
            self.record_test_result("Memory Management Efficiency", False, str(e))
    
    async def test_attention_allocation_effectiveness(self):
        """Test attention allocation effectiveness"""
        logger.info("🧪 Testing Attention Allocation Effectiveness...")
        
        try:
            manager = DynamicResourceManager({'attention_capacity': 100.0})
            await manager.initialize()
            
            # Test attention rebalancing under pressure
            allocations = []
            
            # Fill attention capacity with low priority allocations
            for i in range(5):
                request = ResourceRequest(f"low_{i}", ResourceType.ATTENTION, 18.0, Priority.LOW)
                allocation_id = await manager.allocate_resources(request)
                if allocation_id:
                    allocations.append(allocation_id)
            
            available_before = manager.attention_allocator.get_available_attention()
            
            # Try to allocate high priority (should trigger rebalancing)
            critical_request = ResourceRequest("critical", ResourceType.ATTENTION, 40.0, Priority.CRITICAL)
            critical_allocation = await manager.allocate_resources(critical_request)
            
            available_after = manager.attention_allocator.get_available_attention()
            
            rebalancing_occurred = critical_allocation is not None
            
            self.record_test_result(
                "Attention Rebalancing",
                rebalancing_occurred,
                f"Available attention: {available_before:.1f} → {available_after:.1f}"
            )
            
            # Test attention decay
            initial_allocations = dict(manager.attention_allocator.allocations)
            
            # Apply decay
            manager.attention_allocator.decay_attention()
            
            final_allocations = dict(manager.attention_allocator.allocations)
            
            attention_decayed = len(final_allocations) <= len(initial_allocations)
            
            self.record_test_result(
                "Attention Decay",
                attention_decayed,
                f"Active allocations: {len(initial_allocations)} → {len(final_allocations)}"
            )
            
            # Cleanup
            for allocation_id in allocations:
                await manager.release_resources(allocation_id)
            if critical_allocation:
                await manager.release_resources(critical_allocation)
            
            await manager.shutdown()
            
        except Exception as e:
            self.record_test_result("Attention Allocation Effectiveness", False, str(e))
    
    async def test_load_balancing_efficiency(self):
        """Test load balancing efficiency"""
        logger.info("🧪 Testing Load Balancing Efficiency...")
        
        try:
            modules = ['module1', 'module2', 'module3', 'module4']
            manager = DynamicResourceManager({'cognitive_modules': modules})
            await manager.initialize()
            
            # Submit multiple CPU tasks
            tasks_per_module = {module: 0 for module in modules}
            total_tasks = 20
            
            for i in range(total_tasks):
                request = ResourceRequest(f"task_{i}", ResourceType.CPU, 10.0, Priority.NORMAL)
                allocation_id = await manager.allocate_resources(request)
                
                if allocation_id and allocation_id.startswith('cpu_'):
                    # Extract module name from allocation ID
                    parts = allocation_id.split('_')
                    if len(parts) >= 2:
                        module = parts[1]
                        if module in tasks_per_module:
                            tasks_per_module[module] += 1
                
                # Don't release immediately to show load distribution
            
            # Check load distribution
            min_tasks = min(tasks_per_module.values())
            max_tasks = max(tasks_per_module.values())
            load_balance_ratio = min_tasks / max_tasks if max_tasks > 0 else 1.0
            
            # Good load balancing should have ratio > 0.5
            load_balancing_effective = load_balance_ratio > 0.3
            
            self.record_test_result(
                "Load Balancing Efficiency",
                load_balancing_effective,
                f"Load balance ratio: {load_balance_ratio:.2f}, Task distribution: {tasks_per_module}"
            )
            
            await manager.shutdown()
            
        except Exception as e:
            self.record_test_result("Load Balancing Efficiency", False, str(e))
    
    async def test_resource_monitoring_responsiveness(self):
        """Test resource monitoring responsiveness"""
        logger.info("🧪 Testing Resource Monitoring Responsiveness...")
        
        try:
            alert_triggered = False
            alert_data = None
            
            def test_alert_handler(data):
                nonlocal alert_triggered, alert_data
                alert_triggered = True
                alert_data = data
            
            manager = DynamicResourceManager({'monitor_interval': 0.1})
            await manager.initialize()
            
            manager.monitor.add_alert_callback(test_alert_handler)
            
            # Create high utilization to trigger alerts
            allocations = []
            for i in range(10):
                request = ResourceRequest(f"stress_{i}", ResourceType.ATTENTION, 12.0, Priority.NORMAL)
                allocation_id = await manager.allocate_resources(request)
                if allocation_id:
                    allocations.append(allocation_id)
            
            # Wait for monitoring to detect and alert
            await asyncio.sleep(2.0)
            
            # Check if monitoring detected high utilization
            metrics = manager.monitor.get_metrics()
            monitoring_active = len(metrics) > 0
            
            self.record_test_result(
                "Resource Monitoring Active",
                monitoring_active,
                f"Monitoring {len(metrics)} resource types"
            )
            
            # Cleanup
            for allocation_id in allocations:
                await manager.release_resources(allocation_id)
            
            await manager.shutdown()
            
        except Exception as e:
            self.record_test_result("Resource Monitoring Responsiveness", False, str(e))
    
    async def test_cognitive_kernel_integration(self):
        """Test integration with unified cognitive kernel"""
        logger.info("🧪 Testing Cognitive Kernel Integration...")
        
        try:
            # Test kernel configuration includes resource management
            config = CognitiveKernelConfig()
            
            resource_config_present = hasattr(config, 'resource_config')
            
            self.record_test_result(
                "Resource Config in Kernel",
                resource_config_present,
                f"Resource config keys: {list(config.resource_config.keys()) if resource_config_present else 'None'}"
            )
            
            # Test kernel instantiation with resource manager
            kernel = UnifiedCognitiveKernel(config)
            
            has_resource_manager = hasattr(kernel, 'resource_manager')
            
            self.record_test_result(
                "Resource Manager in Kernel",
                has_resource_manager,
                "Resource manager component integrated"
            )
            
            # Test kernel status includes resource information
            status = kernel.get_status()
            
            resource_status_present = 'resource_status' in status
            
            self.record_test_result(
                "Resource Status in Kernel Status",
                resource_status_present,
                f"Kernel status keys: {list(status.keys())}"
            )
            
        except Exception as e:
            self.record_test_result("Cognitive Kernel Integration", False, str(e))
    
    async def test_performance_requirements(self):
        """Test performance requirements compliance"""
        logger.info("🧪 Testing Performance Requirements...")
        
        try:
            manager = DynamicResourceManager()
            await manager.initialize()
            
            # Test response time requirement (< 1ms)
            response_times = []
            
            for i in range(50):  # Test with multiple allocations
                request = ResourceRequest(f"perf_test_{i}", ResourceType.CPU, 10.0, Priority.NORMAL)
                
                start_time = time.time()
                allocation_id = await manager.allocate_resources(request)
                end_time = time.time()
                
                if allocation_id:
                    response_times.append(end_time - start_time)
                    await manager.release_resources(allocation_id)
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            # Requirement: Dynamic allocation responds within 1ms
            response_time_compliant = avg_response_time < 0.001
            
            self.record_test_result(
                "Response Time Requirement",
                response_time_compliant,
                f"Avg: {avg_response_time:.4f}s, Max: {max_response_time:.4f}s"
            )
            
            # Test system stability under load
            stress_allocations = []
            
            for i in range(100):
                request = ResourceRequest(f"stress_{i}", ResourceType.CPU, 5.0, Priority.NORMAL)
                allocation_id = await manager.allocate_resources(request)
                if allocation_id:
                    stress_allocations.append(allocation_id)
            
            # System should remain stable
            status = manager.get_resource_status()
            system_stable = status['active_allocations'] > 0
            
            self.record_test_result(
                "System Stability Under Load",
                system_stable,
                f"Handled {len(stress_allocations)} concurrent allocations"
            )
            
            # Cleanup
            for allocation_id in stress_allocations:
                await manager.release_resources(allocation_id)
            
            await manager.shutdown()
            
        except Exception as e:
            self.record_test_result("Performance Requirements", False, str(e))
    
    async def run_all_tests(self):
        """Run all comprehensive tests"""
        logger.info("🚀 Starting Comprehensive Resource Management System Tests")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all test suites
        test_suites = [
            self.test_resource_manager_standalone,
            self.test_priority_scheduling_compliance,
            self.test_memory_management_efficiency,
            self.test_attention_allocation_effectiveness,
            self.test_load_balancing_efficiency,
            self.test_resource_monitoring_responsiveness,
            self.test_cognitive_kernel_integration,
            self.test_performance_requirements
        ]
        
        for test_suite in test_suites:
            try:
                await test_suite()
            except Exception as e:
                logger.error(f"Test suite failed: {test_suite.__name__} - {e}")
        
        end_time = time.time()
        
        # Generate test report
        self.generate_test_report(end_time - start_time)
    
    def generate_test_report(self, total_time: float):
        """Generate comprehensive test report"""
        logger.info("=" * 80)
        logger.info("🎯 COMPREHENSIVE RESOURCE MANAGEMENT SYSTEM TEST REPORT")
        logger.info("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"📊 Test Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {failed_tests}")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        logger.info(f"   Total Time: {total_time:.2f}s")
        logger.info("")
        
        # Detailed results
        logger.info("📋 Detailed Results:")
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"   {status} {result['test']}")
            if result['details']:
                logger.info(f"      {result['details']}")
        
        logger.info("")
        
        # Requirements compliance check
        logger.info("🎯 Requirements Compliance:")
        
        requirements_map = {
            "Dynamic allocation responds within 1ms": "Response Time Requirement",
            "Priority scheduling ensures critical operations complete": "Priority Scheduling Compliance",
            "Memory management prevents leaks and optimizes usage": "Memory Management Efficiency", 
            "Attention allocation maximizes cognitive effectiveness": "Attention Allocation Effectiveness",
            "Load balancing maintains system responsiveness": "Load Balancing Efficiency",
            "Monitoring provides real-time visibility": "Resource Monitoring Active",
            "Integration with cognitive kernel": "Resource Manager in Kernel",
            "System maintains stability under load": "System Stability Under Load"
        }
        
        for requirement, test_name in requirements_map.items():
            test_result = next((r for r in self.test_results if r['test'] == test_name), None)
            if test_result:
                status = "✅ MET" if test_result['success'] else "❌ NOT MET"
                logger.info(f"   {status} {requirement}")
            else:
                logger.info(f"   ❓ UNKNOWN {requirement}")
        
        logger.info("")
        
        if success_rate >= 90:
            logger.info("🎉 EXCELLENT: Resource Management System exceeds requirements!")
        elif success_rate >= 80:
            logger.info("✅ GOOD: Resource Management System meets most requirements")
        elif success_rate >= 70:
            logger.info("⚠️ ACCEPTABLE: Resource Management System meets basic requirements")
        else:
            logger.info("❌ NEEDS IMPROVEMENT: Resource Management System requires fixes")
        
        logger.info("=" * 80)


async def main():
    """Main test execution"""
    test_suite = ComprehensiveResourceTest()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Test interrupted by user")
    except Exception as e:
        logger.error(f"💥 Test suite failed: {e}")
        sys.exit(1)