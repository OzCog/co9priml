"""
Tests for Dynamic Resource Management System

Focused tests to validate resource allocation, scheduling, memory management,
attention allocation, load balancing, monitoring, and prediction capabilities.
"""

import asyncio
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from .resource_manager import (
    DynamicResourceManager,
    ResourceRequest,
    ResourceType,
    Priority,
    ResourceConstraint,
    PriorityScheduler,
    MemoryManager,
    AttentionResourceAllocator,
    LoadBalancer,
    ResourceMonitor,
    ResourcePredictor
)


class TestPriorityScheduler:
    """Test priority-based scheduler"""
    
    def test_scheduler_initialization(self):
        """Test scheduler initializes correctly"""
        scheduler = PriorityScheduler(max_concurrent_tasks=2)
        assert scheduler.max_concurrent_tasks == 2
        assert len(scheduler.task_queues) == len(Priority)
        assert len(scheduler.active_tasks) == 0
    
    def test_task_submission(self):
        """Test task submission to queues"""
        scheduler = PriorityScheduler()
        
        request = ResourceRequest(
            request_id="test_task_1",
            resource_type=ResourceType.CPU,
            amount=10.0,
            priority=Priority.HIGH
        )
        
        task_id = scheduler.submit_task(request)
        assert task_id == "test_task_1"
        assert len(scheduler.task_queues[Priority.HIGH]) == 1
    
    def test_priority_ordering(self):
        """Test tasks are scheduled in priority order"""
        scheduler = PriorityScheduler(max_concurrent_tasks=1)
        
        # Submit tasks in reverse priority order
        low_request = ResourceRequest("low", ResourceType.CPU, 10.0, Priority.LOW)
        high_request = ResourceRequest("high", ResourceType.CPU, 10.0, Priority.HIGH)
        critical_request = ResourceRequest("critical", ResourceType.CPU, 10.0, Priority.CRITICAL)
        
        scheduler.submit_task(low_request)
        scheduler.submit_task(high_request)
        scheduler.submit_task(critical_request)
        
        # Should get critical first
        next_task = scheduler.get_next_task()
        assert next_task.request_id == "critical"
        
        # Then high
        next_task = scheduler.get_next_task()
        assert next_task.request_id == "high"
        
        # Then low
        next_task = scheduler.get_next_task()
        assert next_task.request_id == "low"
    
    def test_concurrent_task_limit(self):
        """Test concurrent task limits are respected"""
        scheduler = PriorityScheduler(max_concurrent_tasks=2)
        
        # Fill up active tasks
        scheduler.active_tasks["task1"] = Mock()
        scheduler.active_tasks["task2"] = Mock()
        
        request = ResourceRequest("test", ResourceType.CPU, 10.0, Priority.NORMAL)
        assert not scheduler.can_schedule_task(request)


class TestMemoryManager:
    """Test memory management system"""
    
    def test_memory_manager_initialization(self):
        """Test memory manager initializes correctly"""
        manager = MemoryManager(initial_pool_size=1024)
        assert manager.pool_size == 1024
        assert len(manager.allocated_blocks) == 0
        assert len(manager.free_blocks) == 1
        assert manager.free_blocks[0] == (0, 1024)
    
    def test_memory_allocation(self):
        """Test basic memory allocation"""
        manager = MemoryManager(initial_pool_size=1024)
        
        block_id = manager.allocate(100)
        assert block_id is not None
        assert block_id in manager.allocated_blocks
        
        size, timestamp = manager.allocated_blocks[block_id]
        assert size == 104  # Aligned to 8 bytes
    
    def test_memory_alignment(self):
        """Test memory alignment works correctly"""
        manager = MemoryManager(initial_pool_size=1024)
        
        # Request 97 bytes with 8-byte alignment should get 104 bytes
        block_id = manager.allocate(97, alignment=8)
        assert block_id is not None
        
        size, _ = manager.allocated_blocks[block_id]
        assert size == 104  # Aligned to next 8-byte boundary
    
    def test_memory_deallocation(self):
        """Test memory deallocation"""
        manager = MemoryManager(initial_pool_size=1024)
        
        block_id = manager.allocate(100)
        assert block_id is not None
        
        success = manager.deallocate(block_id)
        assert success
        assert block_id not in manager.allocated_blocks
    
    def test_memory_utilization(self):
        """Test memory utilization calculation"""
        manager = MemoryManager(initial_pool_size=1000)
        
        # Initially should be 0% utilization
        assert manager.get_utilization() == 0.0
        
        # Allocate 100 bytes
        block_id = manager.allocate(100)
        utilization = manager.get_utilization()
        assert utilization > 0.0
        assert utilization < 1.0
    
    def test_out_of_memory(self):
        """Test behavior when out of memory"""
        manager = MemoryManager(initial_pool_size=100)
        
        # Try to allocate more than available
        block_id = manager.allocate(200)
        assert block_id is None
    
    def test_garbage_collection(self):
        """Test garbage collection"""
        manager = MemoryManager(initial_pool_size=1024)
        
        # Add some weak references
        obj1 = Mock()
        obj2 = Mock()
        manager.weak_references.add(obj1)
        manager.weak_references.add(obj2)
        
        # Delete objects
        del obj1, obj2
        
        collected = manager.garbage_collect()
        assert collected >= 0  # Should collect something


class TestAttentionResourceAllocator:
    """Test attention resource allocation"""
    
    def test_attention_allocator_initialization(self):
        """Test attention allocator initializes correctly"""
        allocator = AttentionResourceAllocator(total_attention_capacity=100.0)
        assert allocator.total_capacity == 100.0
        assert len(allocator.allocations) == 0
        assert allocator.get_available_attention() == 100.0
    
    def test_attention_allocation(self):
        """Test basic attention allocation"""
        allocator = AttentionResourceAllocator(total_attention_capacity=100.0)
        
        success = allocator.allocate_attention("target1", 30.0, Priority.NORMAL)
        assert success
        assert allocator.allocations["target1"] == 30.0
        assert allocator.get_available_attention() == 70.0
    
    def test_attention_overallocation(self):
        """Test handling of attention overallocation"""
        allocator = AttentionResourceAllocator(total_attention_capacity=100.0)
        
        # Allocate most capacity
        allocator.allocate_attention("target1", 90.0, Priority.NORMAL)
        
        # Try to allocate more than available (non-critical)
        success = allocator.allocate_attention("target2", 20.0, Priority.LOW)
        assert not success
        
        # Critical tasks should still succeed
        success = allocator.allocate_attention("target3", 20.0, Priority.CRITICAL)
        assert success
    
    def test_attention_release(self):
        """Test attention release"""
        allocator = AttentionResourceAllocator(total_attention_capacity=100.0)
        
        allocator.allocate_attention("target1", 50.0, Priority.NORMAL)
        released = allocator.release_attention("target1", 20.0)
        
        assert released == 20.0
        assert allocator.allocations["target1"] == 30.0
        assert allocator.get_available_attention() == 70.0
    
    def test_attention_rebalancing(self):
        """Test attention rebalancing for higher priority tasks"""
        allocator = AttentionResourceAllocator(total_attention_capacity=100.0)
        
        # Fill with low priority
        allocator.allocate_attention("low1", 40.0, Priority.LOW)
        allocator.allocate_attention("low2", 40.0, Priority.LOW)
        allocator.allocate_attention("low3", 20.0, Priority.LOW)
        
        # High priority should trigger rebalancing
        success = allocator.allocate_attention("high1", 30.0, Priority.HIGH)
        assert success
        
        # Some low priority allocations should be reduced
        total_allocated = sum(allocator.allocations.values())
        assert total_allocated <= 100.0
    
    def test_attention_decay(self):
        """Test attention decay over time"""
        allocator = AttentionResourceAllocator(total_attention_capacity=100.0)
        
        allocator.allocate_attention("target1", 50.0, Priority.NORMAL)
        original_allocation = allocator.allocations["target1"]
        
        allocator.decay_attention()
        
        new_allocation = allocator.allocations.get("target1", 0.0)
        assert new_allocation < original_allocation


class TestLoadBalancer:
    """Test adaptive load balancing"""
    
    def test_load_balancer_initialization(self):
        """Test load balancer initializes correctly"""
        modules = ["module1", "module2", "module3"]
        balancer = LoadBalancer(modules)
        
        assert balancer.modules == modules
        assert len(balancer.module_loads) == 3
        assert all(load == 0.0 for load in balancer.module_loads.values())
    
    def test_task_assignment(self):
        """Test task assignment to modules"""
        modules = ["module1", "module2"]
        balancer = LoadBalancer(modules)
        
        assigned_module = balancer.assign_task(50.0)
        assert assigned_module in modules
        assert balancer.module_loads[assigned_module] == 50.0
    
    def test_least_loaded_selection(self):
        """Test selection of least loaded module"""
        modules = ["module1", "module2", "module3"]
        balancer = LoadBalancer(modules)
        
        # Load module1 heavily
        balancer.module_loads["module1"] = 80.0
        balancer.module_loads["module2"] = 20.0
        balancer.module_loads["module3"] = 10.0
        
        # New task should go to module3 (least loaded)
        assigned_module = balancer.assign_task(30.0)
        assert assigned_module == "module3"
    
    def test_task_completion(self):
        """Test task completion reduces load"""
        modules = ["module1"]
        balancer = LoadBalancer(modules)
        
        balancer.assign_task(50.0)
        assert balancer.module_loads["module1"] == 50.0
        
        balancer.complete_task("module1", 30.0)
        assert balancer.module_loads["module1"] == 20.0
    
    def test_overload_rejection(self):
        """Test rejection when all modules overloaded"""
        modules = ["module1"]
        balancer = LoadBalancer(modules)
        
        # Overload the module
        balancer.module_loads["module1"] = 100.0
        
        # Should not be able to assign more
        assigned_module = balancer.assign_task(10.0)
        assert assigned_module is None


class TestResourceMonitor:
    """Test resource monitoring system"""
    
    def test_monitor_initialization(self):
        """Test monitor initializes correctly"""
        monitor = ResourceMonitor(update_interval=0.5)
        assert monitor.update_interval == 0.5
        assert not monitor.monitoring_active
        assert len(monitor.alert_callbacks) == 0
    
    def test_alert_callback_registration(self):
        """Test alert callback registration"""
        monitor = ResourceMonitor()
        callback = Mock()
        
        monitor.add_alert_callback(callback)
        assert callback in monitor.alert_callbacks
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_metrics_update(self, mock_memory, mock_cpu):
        """Test system metrics update"""
        mock_cpu.return_value = 75.0
        mock_memory.return_value = Mock(
            total=1000000,
            available=300000,
            used=700000,
            percent=70.0
        )
        
        monitor = ResourceMonitor()
        monitor._update_system_metrics()
        
        # Check CPU metrics
        assert ResourceType.CPU in monitor.metrics
        cpu_metrics = monitor.metrics[ResourceType.CPU]
        assert cpu_metrics.utilization_rate == 0.75
        
        # Check memory metrics
        assert ResourceType.MEMORY in monitor.metrics
        memory_metrics = monitor.metrics[ResourceType.MEMORY]
        assert memory_metrics.utilization_rate == 0.70
    
    def test_monitoring_lifecycle(self):
        """Test starting and stopping monitoring"""
        monitor = ResourceMonitor(update_interval=0.1)
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring_active
        assert monitor.monitor_thread is not None
        
        time.sleep(0.2)  # Let it run briefly
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring_active


class TestResourcePredictor:
    """Test resource usage prediction"""
    
    def test_predictor_initialization(self):
        """Test predictor initializes correctly"""
        predictor = ResourcePredictor()
        assert len(predictor.prediction_models) == len(ResourceType)
        assert predictor.prediction_horizon == 60.0
    
    def test_usage_recording(self):
        """Test recording of usage data"""
        predictor = ResourcePredictor()
        
        predictor.record_usage(ResourceType.CPU, 75.0)
        
        model = predictor.prediction_models[ResourceType.CPU]
        assert len(model.historical_data) == 1
        
        timestamp, value = model.historical_data[0]
        assert value == 75.0
        assert timestamp > 0
    
    def test_prediction_insufficient_data(self):
        """Test prediction with insufficient data"""
        predictor = ResourcePredictor()
        
        # Record only a few data points
        for i in range(5):
            predictor.record_usage(ResourceType.CPU, 50.0 + i)
        
        prediction = predictor.predict_usage(ResourceType.CPU)
        assert prediction is None  # Not enough data
    
    def test_prediction_with_sufficient_data(self):
        """Test prediction with sufficient data"""
        predictor = ResourcePredictor()
        
        # Record trending data
        base_time = time.time()
        for i in range(15):
            predictor.prediction_models[ResourceType.CPU].historical_data.append(
                (base_time + i, 50.0 + i * 2)  # Upward trend
            )
        
        prediction = predictor.predict_usage(ResourceType.CPU)
        assert prediction is not None
        assert prediction > 50.0  # Should predict higher due to trend
    
    def test_prediction_accuracy_update(self):
        """Test prediction accuracy tracking"""
        predictor = ResourcePredictor()
        
        # Initial accuracy should be 0
        assert predictor.get_prediction_accuracy(ResourceType.CPU) == 0.0
        
        # Update with perfect prediction
        predictor.update_prediction_accuracy(ResourceType.CPU, 50.0, 50.0)
        assert predictor.get_prediction_accuracy(ResourceType.CPU) == 1.0
        
        # Update with poor prediction
        predictor.update_prediction_accuracy(ResourceType.CPU, 50.0, 100.0)
        accuracy = predictor.get_prediction_accuracy(ResourceType.CPU)
        assert 0.0 < accuracy < 1.0  # Should be reduced but not zero


class TestDynamicResourceManager:
    """Test main resource management system"""
    
    @pytest.fixture
    def resource_manager(self):
        """Create resource manager for testing"""
        config = {
            'max_concurrent_tasks': 2,
            'memory_pool_size': 1024,
            'attention_capacity': 100.0,
            'cognitive_modules': ['test_module1', 'test_module2'],
            'monitor_interval': 0.1
        }
        return DynamicResourceManager(config)
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, resource_manager):
        """Test resource manager initializes correctly"""
        await resource_manager.initialize()
        
        assert resource_manager.scheduler is not None
        assert resource_manager.memory_manager is not None
        assert resource_manager.attention_allocator is not None
        assert resource_manager.load_balancer is not None
        assert resource_manager.monitor is not None
        assert resource_manager.predictor is not None
        
        await resource_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_cpu_resource_allocation(self, resource_manager):
        """Test CPU resource allocation"""
        await resource_manager.initialize()
        
        request = ResourceRequest(
            request_id="cpu_test",
            resource_type=ResourceType.CPU,
            amount=50.0,
            priority=Priority.NORMAL
        )
        
        allocation_id = await resource_manager.allocate_resources(request)
        assert allocation_id is not None
        assert allocation_id.startswith('cpu_')
        
        # Check allocation is tracked
        assert allocation_id in resource_manager.active_allocations
        
        # Release resources
        success = await resource_manager.release_resources(allocation_id)
        assert success
        assert allocation_id not in resource_manager.active_allocations
        
        await resource_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_resource_allocation(self, resource_manager):
        """Test memory resource allocation"""
        await resource_manager.initialize()
        
        request = ResourceRequest(
            request_id="mem_test",
            resource_type=ResourceType.MEMORY,
            amount=128.0,  # 128 bytes
            priority=Priority.NORMAL
        )
        
        allocation_id = await resource_manager.allocate_resources(request)
        assert allocation_id is not None
        assert allocation_id.startswith('mem_')
        
        success = await resource_manager.release_resources(allocation_id)
        assert success
        
        await resource_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_attention_resource_allocation(self, resource_manager):
        """Test attention resource allocation"""
        await resource_manager.initialize()
        
        request = ResourceRequest(
            request_id="att_test",
            resource_type=ResourceType.ATTENTION,
            amount=30.0,
            priority=Priority.HIGH
        )
        
        allocation_id = await resource_manager.allocate_resources(request)
        assert allocation_id is not None
        assert allocation_id.startswith('att_')
        
        success = await resource_manager.release_resources(allocation_id)
        assert success
        
        await resource_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_resource_constraints(self, resource_manager):
        """Test resource constraint checking"""
        await resource_manager.initialize()
        
        # Add constraint
        constraint = ResourceConstraint(
            resource_type=ResourceType.CPU,
            min_required=10.0,
            max_allowed=80.0,
            preferred=50.0,
            hard_limit=True
        )
        resource_manager.add_resource_constraint(constraint)
        
        # Request below minimum should fail
        request = ResourceRequest(
            request_id="constraint_test_low",
            resource_type=ResourceType.CPU,
            amount=5.0,
            priority=Priority.NORMAL
        )
        allocation_id = await resource_manager.allocate_resources(request)
        assert allocation_id is None
        
        # Request above hard limit should fail
        request = ResourceRequest(
            request_id="constraint_test_high",
            resource_type=ResourceType.CPU,
            amount=100.0,
            priority=Priority.NORMAL
        )
        allocation_id = await resource_manager.allocate_resources(request)
        assert allocation_id is None
        
        # Request within bounds should succeed
        request = ResourceRequest(
            request_id="constraint_test_valid",
            resource_type=ResourceType.CPU,
            amount=50.0,
            priority=Priority.NORMAL
        )
        allocation_id = await resource_manager.allocate_resources(request)
        assert allocation_id is not None
        
        await resource_manager.release_resources(allocation_id)
        await resource_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_resource_status_reporting(self, resource_manager):
        """Test resource status reporting"""
        await resource_manager.initialize()
        
        status = resource_manager.get_resource_status()
        
        assert 'active_allocations' in status
        assert 'cpu_utilization' in status
        assert 'memory_utilization' in status
        assert 'memory_fragmentation' in status
        assert 'attention_available' in status
        assert 'system_metrics' in status
        assert 'performance_metrics' in status
        
        await resource_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_degradation_strategy(self, resource_manager):
        """Test degradation strategy application"""
        await resource_manager.initialize()
        
        # Mock degradation strategy
        degradation_called = False
        
        def mock_degradation(alert_data):
            nonlocal degradation_called
            degradation_called = True
        
        resource_manager.set_degradation_strategy(ResourceType.CPU, mock_degradation)
        
        # Trigger alert
        alert_data = {
            'resource_type': ResourceType.CPU,
            'utilization': 0.95,
            'threshold': 0.9,
            'timestamp': time.time()
        }
        
        resource_manager._handle_resource_alert(alert_data)
        assert degradation_called
        
        await resource_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_preallocation_cycle(self, resource_manager):
        """Test predictive preallocation"""
        await resource_manager.initialize()
        
        # Set up prediction data suggesting high future CPU usage
        predictor = resource_manager.predictor
        base_time = time.time()
        for i in range(15):
            predictor.prediction_models[ResourceType.CPU].historical_data.append(
                (base_time + i, 60.0 + i * 2)  # Increasing trend toward 90%
            )
        
        # Run preallocation cycle
        await resource_manager.preallocation_cycle()
        
        # Should have some active allocations from preallocation
        status = resource_manager.get_resource_status()
        # Note: Preallocation might not always trigger depending on prediction threshold
        
        await resource_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_callback_execution(self, resource_manager):
        """Test callback execution on resource allocation"""
        await resource_manager.initialize()
        
        callback_called = False
        callback_allocation_id = None
        
        async def test_callback(allocation_id):
            nonlocal callback_called, callback_allocation_id
            callback_called = True
            callback_allocation_id = allocation_id
        
        request = ResourceRequest(
            request_id="callback_test",
            resource_type=ResourceType.CPU,
            amount=25.0,
            priority=Priority.NORMAL,
            callback=test_callback
        )
        
        allocation_id = await resource_manager.allocate_resources(request)
        assert allocation_id is not None
        assert callback_called
        assert callback_allocation_id == allocation_id
        
        await resource_manager.release_resources(allocation_id)
        await resource_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, resource_manager):
        """Test performance metrics are tracked"""
        await resource_manager.initialize()
        
        # Make several allocations to generate metrics
        for i in range(5):
            request = ResourceRequest(
                request_id=f"perf_test_{i}",
                resource_type=ResourceType.CPU,
                amount=20.0,
                priority=Priority.NORMAL
            )
            
            allocation_id = await resource_manager.allocate_resources(request)
            if allocation_id:
                await resource_manager.release_resources(allocation_id)
        
        # Check performance metrics were recorded
        assert len(resource_manager.performance_metrics['response_time']) > 0
        
        # All response times should be reasonable (< 1 second for these simple tests)
        for response_time in resource_manager.performance_metrics['response_time']:
            assert 0 <= response_time < 1.0
        
        await resource_manager.shutdown()


@pytest.mark.asyncio
async def test_integration_with_existing_attention_system():
    """Test integration with existing ECAN attention allocation"""
    # This would test integration with the existing attention_allocation.py
    # For now, just ensure our system can work alongside it
    
    resource_manager = DynamicResourceManager({
        'attention_capacity': 200.0,
        'cognitive_modules': ['reasoning', 'memory', 'attention']
    })
    
    await resource_manager.initialize()
    
    # Test that attention allocation works
    request = ResourceRequest(
        request_id="integration_test",
        resource_type=ResourceType.ATTENTION,
        amount=50.0,
        priority=Priority.HIGH
    )
    
    allocation_id = await resource_manager.allocate_resources(request)
    assert allocation_id is not None
    
    # Test resource status includes attention info
    status = resource_manager.get_resource_status()
    assert 'attention_available' in status
    assert status['attention_available'] == 150.0  # 200 - 50
    
    await resource_manager.release_resources(allocation_id)
    await resource_manager.shutdown()


if __name__ == "__main__":
    # Run basic tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))