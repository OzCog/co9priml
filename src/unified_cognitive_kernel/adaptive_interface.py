"""
Adaptive Interface Layer

Implements the unified API gateway and embodied agents interface for
the cognitive kernel, providing coherent access to all cognitive components.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

# For API implementation
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# For gRPC implementation
try:
    import grpc
    from concurrent import futures
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False


class InterfaceType(Enum):
    """Types of adaptive interfaces"""
    REST_API = "rest_api"
    GRPC_API = "grpc_api"
    WEBSOCKET = "websocket"
    EMBODIED_AGENT = "embodied_agent"
    VIRTUAL_INTERFACE = "virtual_interface"


class AgentType(Enum):
    """Types of embodied agents"""
    CHATBOT = "chatbot"
    ROBOT = "robot"
    GAME_CHARACTER = "game_character"
    VIRTUAL_ASSISTANT = "virtual_assistant"
    COGNITIVE_TUTOR = "cognitive_tutor"


@dataclass
class InterfaceRequest:
    """Request to the adaptive interface"""
    request_id: str
    interface_type: InterfaceType
    agent_type: Optional[AgentType] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterfaceResponse:
    """Response from the adaptive interface"""
    request_id: str
    success: bool
    output_data: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    cognitive_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbodiedAgent:
    """Embodied agent configuration"""
    agent_id: str
    agent_type: AgentType
    capabilities: List[str]
    sensory_modalities: List[str]
    action_repertoire: List[str]
    learning_enabled: bool = True
    meta_cognitive_enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


class AdaptiveInterface(ABC):
    """Abstract base class for adaptive interfaces"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the interface"""
        pass
    
    @abstractmethod
    async def process_request(self, request: InterfaceRequest) -> InterfaceResponse:
        """Process an interface request"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the interface"""
        pass


class RESTAPIInterface(AdaptiveInterface):
    """REST API interface implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.app = FastAPI(title="Unified Cognitive Kernel API") if FASTAPI_AVAILABLE else None
        self.port = config.get('port', 8080)
        self.host = config.get('host', '0.0.0.0')
        self.cognitive_kernel = None
        
        if self.app:
            self._setup_routes()
    
    def _setup_routes(self):
        """Setup REST API routes"""
        if not self.app:
            return
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.app.get("/")
        async def root():
            return {"message": "Unified Cognitive Kernel API", "version": "0.1.0"}
        
        @self.app.get("/status")
        async def get_status():
            if self.cognitive_kernel:
                return self.cognitive_kernel.get_status()
            return {"status": "kernel not initialized"}
        
        @self.app.post("/cognitive_cycle")
        async def cognitive_cycle(request_data: dict):
            if not self.cognitive_kernel:
                raise HTTPException(status_code=503, detail="Kernel not initialized")
            
            try:
                response = await self.cognitive_kernel.cognitive_cycle(request_data)
                return response
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/tensor_operation")
        async def tensor_operation(operation_data: dict):
            if not self.cognitive_kernel:
                raise HTTPException(status_code=503, detail="Kernel not initialized")
            
            try:
                response = await self.cognitive_kernel.tensor_kernel.process_input(operation_data)
                return response
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/reasoning")
        async def reasoning(reasoning_data: dict):
            if not self.cognitive_kernel:
                raise HTTPException(status_code=503, detail="Kernel not initialized")
            
            try:
                response = await self.cognitive_kernel.cognitive_grammar.process_reasoning(
                    {}, reasoning_data
                )
                return response
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/attention_allocation")
        async def attention_allocation(attention_data: dict):
            if not self.cognitive_kernel:
                raise HTTPException(status_code=503, detail="Kernel not initialized")
            
            try:
                response = await self.cognitive_kernel.attention_allocation.allocate_attention(
                    attention_data, {}
                )
                return response
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_text()
                    request_data = json.loads(data)
                    
                    if self.cognitive_kernel:
                        response = await self.cognitive_kernel.cognitive_cycle(request_data)
                        await websocket.send_text(json.dumps(response))
                    else:
                        await websocket.send_text(json.dumps({"error": "Kernel not initialized"}))
                        
            except WebSocketDisconnect:
                pass
    
    async def initialize(self) -> None:
        """Initialize REST API interface"""
        if not FASTAPI_AVAILABLE:
            self.logger.warning("FastAPI not available, REST API interface disabled")
            return
        
        self.logger.info(f"Starting REST API interface on {self.host}:{self.port}")
        
        # Start the server in a separate task
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        self.server = uvicorn.Server(config)
        
        # Start server in background
        asyncio.create_task(self.server.serve())
        
        self.logger.info("REST API interface initialized")
    
    async def process_request(self, request: InterfaceRequest) -> InterfaceResponse:
        """Process REST API request"""
        # This is handled by the FastAPI routes
        return InterfaceResponse(
            request_id=request.request_id,
            success=True,
            output_data={"message": "Request processed via REST API"}
        )
    
    def set_cognitive_kernel(self, kernel):
        """Set reference to cognitive kernel"""
        self.cognitive_kernel = kernel
    
    async def shutdown(self) -> None:
        """Shutdown REST API interface"""
        if hasattr(self, 'server'):
            self.server.should_exit = True
        self.logger.info("REST API interface shutdown complete")


class GRPCAPIInterface(AdaptiveInterface):
    """gRPC API interface implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.port = config.get('grpc_port', 50051)
        self.server = None
        self.cognitive_kernel = None
    
    async def initialize(self) -> None:
        """Initialize gRPC interface"""
        if not GRPC_AVAILABLE:
            self.logger.warning("gRPC not available, gRPC API interface disabled")
            return
        
        self.logger.info(f"Starting gRPC interface on port {self.port}")
        
        # Setup gRPC server
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # Add service implementations here
        # self.server.add_insecure_port(f'[::]:{self.port}')
        
        self.logger.info("gRPC interface initialized")
    
    async def process_request(self, request: InterfaceRequest) -> InterfaceResponse:
        """Process gRPC request"""
        return InterfaceResponse(
            request_id=request.request_id,
            success=True,
            output_data={"message": "Request processed via gRPC"}
        )
    
    def set_cognitive_kernel(self, kernel):
        """Set reference to cognitive kernel"""
        self.cognitive_kernel = kernel
    
    async def shutdown(self) -> None:
        """Shutdown gRPC interface"""
        if self.server:
            await self.server.stop(0)
        self.logger.info("gRPC interface shutdown complete")


class EmbodiedAgentInterface(AdaptiveInterface):
    """Embodied agent interface implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.agents = {}
        self.cognitive_kernel = None
        self.active_sessions = {}
    
    async def initialize(self) -> None:
        """Initialize embodied agent interface"""
        self.logger.info("Initializing embodied agent interface...")
        
        # Create default agents
        default_agents = [
            EmbodiedAgent(
                agent_id="chatbot_1",
                agent_type=AgentType.CHATBOT,
                capabilities=["conversation", "question_answering", "reasoning"],
                sensory_modalities=["text", "audio"],
                action_repertoire=["respond", "ask_question", "explain"]
            ),
            EmbodiedAgent(
                agent_id="virtual_assistant_1",
                agent_type=AgentType.VIRTUAL_ASSISTANT,
                capabilities=["task_planning", "information_retrieval", "scheduling"],
                sensory_modalities=["text", "audio", "visual"],
                action_repertoire=["execute_task", "provide_information", "schedule_event"]
            ),
            EmbodiedAgent(
                agent_id="cognitive_tutor_1",
                agent_type=AgentType.COGNITIVE_TUTOR,
                capabilities=["teaching", "assessment", "personalized_learning"],
                sensory_modalities=["text", "visual", "performance_metrics"],
                action_repertoire=["present_material", "assess_understanding", "provide_feedback"]
            )
        ]
        
        for agent in default_agents:
            self.agents[agent.agent_id] = agent
        
        self.logger.info(f"Embodied agent interface initialized with {len(self.agents)} agents")
    
    async def process_request(self, request: InterfaceRequest) -> InterfaceResponse:
        """Process embodied agent request"""
        agent_id = request.input_data.get('agent_id')
        
        if not agent_id or agent_id not in self.agents:
            return InterfaceResponse(
                request_id=request.request_id,
                success=False,
                output_data={"error": "Agent not found"}
            )
        
        agent = self.agents[agent_id]
        
        # Process request through cognitive kernel
        if self.cognitive_kernel:
            cognitive_response = await self.cognitive_kernel.cognitive_cycle(request.input_data)
            
            # Generate agent-specific response
            agent_response = await self._generate_agent_response(agent, cognitive_response, request)
            
            return InterfaceResponse(
                request_id=request.request_id,
                success=True,
                output_data=agent_response,
                cognitive_state=cognitive_response
            )
        
        return InterfaceResponse(
            request_id=request.request_id,
            success=False,
            output_data={"error": "Cognitive kernel not available"}
        )
    
    async def _generate_agent_response(self, agent: EmbodiedAgent, 
                                     cognitive_response: Dict[str, Any],
                                     request: InterfaceRequest) -> Dict[str, Any]:
        """Generate agent-specific response based on cognitive processing"""
        
        # Extract relevant information from cognitive response
        tensor_results = cognitive_response.get('tensor_results', {})
        reasoning_results = cognitive_response.get('reasoning_results', {})
        attention_results = cognitive_response.get('attention_results', {})
        
        # Generate response based on agent type
        if agent.agent_type == AgentType.CHATBOT:
            return await self._generate_chatbot_response(agent, cognitive_response, request)
        elif agent.agent_type == AgentType.VIRTUAL_ASSISTANT:
            return await self._generate_assistant_response(agent, cognitive_response, request)
        elif agent.agent_type == AgentType.COGNITIVE_TUTOR:
            return await self._generate_tutor_response(agent, cognitive_response, request)
        else:
            return await self._generate_generic_response(agent, cognitive_response, request)
    
    async def _generate_chatbot_response(self, agent: EmbodiedAgent,
                                       cognitive_response: Dict[str, Any],
                                       request: InterfaceRequest) -> Dict[str, Any]:
        """Generate chatbot-specific response"""
        user_input = request.input_data.get('message', '')
        
        # Simple response generation based on cognitive processing
        response_text = "I understand you're asking about: " + user_input
        
        # Add reasoning if available
        if 'reasoning_results' in cognitive_response:
            reasoning = cognitive_response['reasoning_results']
            if 'inferences' in reasoning:
                response_text += f" Based on my reasoning, I can infer {len(reasoning['inferences'])} relevant connections."
        
        return {
            'agent_id': agent.agent_id,
            'agent_type': agent.agent_type.value,
            'response': response_text,
            'capabilities_used': ['conversation', 'reasoning'],
            'confidence': 0.8
        }
    
    async def _generate_assistant_response(self, agent: EmbodiedAgent,
                                         cognitive_response: Dict[str, Any],
                                         request: InterfaceRequest) -> Dict[str, Any]:
        """Generate virtual assistant response"""
        task = request.input_data.get('task', '')
        
        response_text = f"I can help you with: {task}"
        actions = []
        
        # Generate task-specific actions
        if 'schedule' in task.lower():
            actions.append({'action': 'schedule_event', 'parameters': {'task': task}})
        elif 'information' in task.lower():
            actions.append({'action': 'retrieve_information', 'parameters': {'query': task}})
        else:
            actions.append({'action': 'execute_task', 'parameters': {'task': task}})
        
        return {
            'agent_id': agent.agent_id,
            'agent_type': agent.agent_type.value,
            'response': response_text,
            'actions': actions,
            'capabilities_used': ['task_planning', 'information_retrieval'],
            'confidence': 0.85
        }
    
    async def _generate_tutor_response(self, agent: EmbodiedAgent,
                                     cognitive_response: Dict[str, Any],
                                     request: InterfaceRequest) -> Dict[str, Any]:
        """Generate cognitive tutor response"""
        learning_content = request.input_data.get('content', '')
        student_response = request.input_data.get('student_response', '')
        
        response_text = f"Let me help you understand: {learning_content}"
        
        # Add assessment if student response provided
        if student_response:
            response_text += f" Your response shows understanding of key concepts."
        
        feedback = {
            'content_mastery': 0.7,
            'areas_for_improvement': ['conceptual_understanding'],
            'next_steps': ['practice_exercises', 'concept_review']
        }
        
        return {
            'agent_id': agent.agent_id,
            'agent_type': agent.agent_type.value,
            'response': response_text,
            'feedback': feedback,
            'capabilities_used': ['teaching', 'assessment'],
            'confidence': 0.9
        }
    
    async def _generate_generic_response(self, agent: EmbodiedAgent,
                                       cognitive_response: Dict[str, Any],
                                       request: InterfaceRequest) -> Dict[str, Any]:
        """Generate generic agent response"""
        return {
            'agent_id': agent.agent_id,
            'agent_type': agent.agent_type.value,
            'response': 'I have processed your request using my cognitive capabilities.',
            'capabilities_used': agent.capabilities,
            'confidence': 0.75
        }
    
    def set_cognitive_kernel(self, kernel):
        """Set reference to cognitive kernel"""
        self.cognitive_kernel = kernel
    
    async def shutdown(self) -> None:
        """Shutdown embodied agent interface"""
        self.agents.clear()
        self.active_sessions.clear()
        self.logger.info("Embodied agent interface shutdown complete")


class AdaptiveInterfaceLayer:
    """
    Unified adaptive interface layer that manages all interface types.
    
    This class provides a coherent API for accessing all cognitive kernel
    components through multiple interface modalities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize interface implementations
        self.interfaces = {}
        self.cognitive_kernel = None
        self.active_sessions = {}
        self.request_history = []
        
        # Component registry
        self.components = {}
        
        self.logger.info("Adaptive interface layer initialized")
    
    async def initialize(self) -> None:
        """Initialize all adaptive interfaces"""
        self.logger.info("Initializing adaptive interface layer...")
        
        try:
            # Initialize REST API interface
            if self.config.get('enable_rest', True):
                rest_interface = RESTAPIInterface(self.config)
                await rest_interface.initialize()
                self.interfaces[InterfaceType.REST_API] = rest_interface
                self.logger.info("REST API interface initialized")
            
            # Initialize gRPC interface
            if self.config.get('enable_grpc', False):
                grpc_interface = GRPCAPIInterface(self.config)
                await grpc_interface.initialize()
                self.interfaces[InterfaceType.GRPC_API] = grpc_interface
                self.logger.info("gRPC interface initialized")
            
            # Initialize embodied agent interface
            agent_interface = EmbodiedAgentInterface(self.config)
            await agent_interface.initialize()
            self.interfaces[InterfaceType.EMBODIED_AGENT] = agent_interface
            self.logger.info("Embodied agent interface initialized")
            
            self.logger.info("Adaptive interface layer fully initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize adaptive interface layer: {e}")
            raise
    
    def register_component(self, name: str, component: Any) -> None:
        """Register a cognitive component"""
        self.components[name] = component
        
        # Set kernel reference in interfaces
        if name == 'cognitive_kernel':
            self.cognitive_kernel = component
            for interface in self.interfaces.values():
                if hasattr(interface, 'set_cognitive_kernel'):
                    interface.set_cognitive_kernel(component)
        
        self.logger.info(f"Registered component: {name}")
    
    async def process_request(self, interface_type: InterfaceType, 
                            request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request through specified interface"""
        if interface_type not in self.interfaces:
            raise ValueError(f"Interface type {interface_type} not available")
        
        interface = self.interfaces[interface_type]
        
        # Create interface request
        request = InterfaceRequest(
            request_id=f"req_{len(self.request_history)}",
            interface_type=interface_type,
            input_data=request_data
        )
        
        # Process request
        response = await interface.process_request(request)
        
        # Track request history
        self.request_history.append({
            'request': request,
            'response': response,
            'timestamp': asyncio.get_event_loop().time()
        })
        
        return response.output_data
    
    async def generate_response(self, attention_response: Dict[str, Any], 
                              gestalt_tensor: np.ndarray) -> Dict[str, Any]:
        """Generate unified response from attention allocation and gestalt tensor"""
        # Combine attention response with gestalt tensor information
        response = {
            'attention_allocation': attention_response,
            'gestalt_tensor_shape': gestalt_tensor.shape,
            'gestalt_tensor_norm': float(np.linalg.norm(gestalt_tensor)),
            'interfaces_available': [interface_type.value for interface_type in self.interfaces.keys()],
            'components_registered': list(self.components.keys()),
            'active_sessions': len(self.active_sessions)
        }
        
        # Add interface-specific enhancements
        if InterfaceType.EMBODIED_AGENT in self.interfaces:
            agent_interface = self.interfaces[InterfaceType.EMBODIED_AGENT]
            response['available_agents'] = list(agent_interface.agents.keys())
        
        return response
    
    async def get_quality_metrics(self) -> Dict[str, float]:
        """Get quality metrics for interface responses"""
        if not self.request_history:
            return {'response_quality': 0.0, 'user_satisfaction': 0.0}
        
        # Calculate metrics based on request history
        recent_requests = self.request_history[-10:]  # Last 10 requests
        
        success_rate = sum(1 for req in recent_requests if req['response'].success) / len(recent_requests)
        response_time = sum(req['response'].metadata.get('processing_time', 0.1) for req in recent_requests) / len(recent_requests)
        
        return {
            'response_quality': success_rate,
            'user_satisfaction': 0.8,  # Placeholder
            'response_time': response_time,
            'interface_utilization': len(self.interfaces) / 4  # Normalized by max interfaces
        }
    
    def is_active(self) -> bool:
        """Check if adaptive interface layer is active"""
        return len(self.interfaces) > 0
    
    async def shutdown(self) -> None:
        """Shutdown adaptive interface layer"""
        self.logger.info("Shutting down adaptive interface layer...")
        
        # Shutdown all interfaces
        for interface in self.interfaces.values():
            await interface.shutdown()
        
        # Clear state
        self.interfaces.clear()
        self.components.clear()
        self.active_sessions.clear()
        self.request_history.clear()
        
        self.logger.info("Adaptive interface layer shutdown complete")