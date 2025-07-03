#include "cognitive-agent.h"
#include <stdio.h>
#include <unistd.h>

// Demo: Simple consciousness exploration task
void demo_consciousness_exploration(void) {
    printf("\n=== Consciousness Exploration Demo ===\n");
    
    // Create two cognitive agents
    cognitive_agent* philosopher = create_cognitive_agent("localhost:8001");
    cognitive_agent* scientist = create_cognitive_agent("localhost:8002");
    
    // Add relevant knowledge to each agent
    printf("\nAdding knowledge to agents...\n");
    
    // Philosopher's knowledge
    float consciousness_emb[64];
    for (int i = 0; i < 64; i++) consciousness_emb[i] = (float)i / 64.0f;
    add_knowledge(philosopher->memory, "consciousness", consciousness_emb, 64);
    
    float philosophy_emb[64];
    for (int i = 0; i < 64; i++) philosophy_emb[i] = 1.0f - (float)i / 64.0f;
    add_knowledge(philosopher->memory, "philosophy_of_mind", philosophy_emb, 64);
    
    // Scientist's knowledge
    float neuroscience_emb[64];
    for (int i = 0; i < 64; i++) neuroscience_emb[i] = sinf((float)i / 64.0f * 3.14159f);
    add_knowledge(scientist->memory, "neuroscience", neuroscience_emb, 64);
    
    float cognition_emb[64];
    for (int i = 0; i < 64; i++) cognition_emb[i] = cosf((float)i / 64.0f * 3.14159f);
    add_knowledge(scientist->memory, "cognitive_science", cognition_emb, 64);
    
    // Simulate consciousness exploration task
    printf("\nSimulating consciousness exploration...\n");
    
    // Philosopher initiates inquiry
    ggml_tensor* inquiry = ggml_new_tensor_1d(philosopher->ctx, GGML_TYPE_F32, 128);
    float* inquiry_data = (float*)inquiry->data;
    for (int i = 0; i < 128; i++) {
        inquiry_data[i] = (float)i / 128.0f;  // Represents "What is consciousness?"
    }
    
    // Philosopher allocates attention to reasoning
    allocate_attention(philosopher->attention, 0.6f, COGNITIVE_TYPE_REASONING);
    
    // Send inquiry to scientist
    send_cognitive_tensor(philosopher, scientist->agent_id, inquiry, 0.8f);
    
    // Scientist processes the inquiry
    cognitive_tensor_packet msg = {0};
    msg.cognitive_type = COGNITIVE_TYPE_REASONING;
    msg.attention_weight = 0.8f;
    msg.source_agent_id = philosopher->agent_id;
    msg.target_agent_id = scientist->agent_id;
    msg.salience_score = 0.9f;
    
    process_incoming_tensor(scientist, &msg);
    
    // Scientist formulates response based on neuroscience knowledge
    ggml_tensor* response = ggml_new_tensor_1d(scientist->ctx, GGML_TYPE_F32, 256);
    float* response_data = (float*)response->data;
    for (int i = 0; i < 256; i++) {
        response_data[i] = sinf((float)i / 256.0f * 6.28f);  // Neural patterns
    }
    
    // Send response back
    send_cognitive_tensor(scientist, philosopher->agent_id, response, 0.7f);
    
    // Philosopher processes scientific perspective
    msg.cognitive_type = COGNITIVE_TYPE_MEMORY;
    msg.attention_weight = 0.7f;
    msg.source_agent_id = scientist->agent_id;
    msg.target_agent_id = philosopher->agent_id;
    
    process_incoming_tensor(philosopher, &msg);
    
    // Both agents update their beliefs through reasoning
    printf("\nCognitive state updates:\n");
    printf("Philosopher - Inferences made: %lu\n", philosopher->reasoning->inferences_made);
    printf("Scientist - Inferences made: %lu\n", scientist->reasoning->inferences_made);
    
    // Search for concepts in memory
    hypergraph_node* consciousness_node = find_concept(philosopher->memory, "consciousness");
    if (consciousness_node) {
        printf("Philosopher found consciousness concept with truth value: %.2f\n", 
               consciousness_node->truth_value);
    }
    
    hypergraph_node* neuroscience_node = find_concept(scientist->memory, "neuroscience");
    if (neuroscience_node) {
        printf("Scientist found neuroscience concept with truth value: %.2f\n", 
               neuroscience_node->truth_value);
    }
    
    // Display attention allocation
    printf("\nAttention allocation summary:\n");
    printf("Philosopher - Reasoning: %.2f, Memory: %.2f, Communication: %.2f\n",
           philosopher->attention->reasoning_allocation,
           philosopher->attention->memory_allocation,
           philosopher->attention->communication_allocation);
    
    printf("Scientist - Reasoning: %.2f, Memory: %.2f, Communication: %.2f\n",
           scientist->attention->reasoning_allocation,
           scientist->attention->memory_allocation,
           scientist->attention->communication_allocation);
    
    // Clean up
    cleanup_cognitive_agent(philosopher);
    cleanup_cognitive_agent(scientist);
    
    printf("\nConsciousness exploration demo completed.\n");
}

// Demo: Distributed problem solving
void demo_distributed_problem_solving(void) {
    printf("\n=== Distributed Problem Solving Demo ===\n");
    
    // Create a network of three agents with different specializations
    cognitive_agent* coordinator = create_cognitive_agent("localhost:9001");
    cognitive_agent* analyzer = create_cognitive_agent("localhost:9002");
    cognitive_agent* synthesizer = create_cognitive_agent("localhost:9003");
    
    printf("\nSpecializing agents...\n");
    
    // Coordinator: Task decomposition specialist
    float planning_emb[32];
    for (int i = 0; i < 32; i++) planning_emb[i] = (float)i / 32.0f;
    add_knowledge(coordinator->memory, "task_planning", planning_emb, 32);
    
    // Analyzer: Pattern recognition specialist
    float analysis_emb[32];
    for (int i = 0; i < 32; i++) analysis_emb[i] = (float)(32-i) / 32.0f;
    add_knowledge(analyzer->memory, "pattern_analysis", analysis_emb, 32);
    
    // Synthesizer: Solution integration specialist
    float synthesis_emb[32];
    for (int i = 0; i < 32; i++) synthesis_emb[i] = sinf((float)i / 32.0f * 3.14159f);
    add_knowledge(synthesizer->memory, "solution_synthesis", synthesis_emb, 32);
    
    // Simulate complex problem requiring distributed cognition
    printf("\nSimulating distributed problem solving...\n");
    
    // Coordinator receives complex problem
    ggml_tensor* complex_problem = ggml_new_tensor_1d(coordinator->ctx, GGML_TYPE_F32, 512);
    
    // Coordinator decomposes problem and allocates attention
    allocate_attention(coordinator->attention, 0.4f, COGNITIVE_TYPE_TASK);
    allocate_attention(coordinator->attention, 0.3f, COGNITIVE_TYPE_COMMUNICATION);
    
    printf("Coordinator decomposing problem...\n");
    
    // Send analysis task to analyzer
    ggml_tensor* analysis_task = ggml_new_tensor_1d(coordinator->ctx, GGML_TYPE_F32, 128);
    send_cognitive_tensor(coordinator, analyzer->agent_id, analysis_task, 0.7f);
    
    // Analyzer processes analysis task
    cognitive_tensor_packet analysis_msg = {0};
    analysis_msg.cognitive_type = COGNITIVE_TYPE_TASK;
    analysis_msg.attention_weight = 0.7f;
    analysis_msg.source_agent_id = coordinator->agent_id;
    analysis_msg.target_agent_id = analyzer->agent_id;
    
    process_incoming_tensor(analyzer, &analysis_msg);
    
    // Analyzer performs analysis and sends results
    ggml_tensor* analysis_results = ggml_new_tensor_1d(analyzer->ctx, GGML_TYPE_F32, 64);
    send_cognitive_tensor(analyzer, synthesizer->agent_id, analysis_results, 0.6f);
    
    // Synthesizer receives analysis and creates solution
    cognitive_tensor_packet synthesis_msg = {0};
    synthesis_msg.cognitive_type = COGNITIVE_TYPE_REASONING;
    synthesis_msg.attention_weight = 0.6f;
    synthesis_msg.source_agent_id = analyzer->agent_id;
    synthesis_msg.target_agent_id = synthesizer->agent_id;
    
    process_incoming_tensor(synthesizer, &synthesis_msg);
    
    // Solution synthesis
    ggml_tensor* solution = ggml_new_tensor_1d(synthesizer->ctx, GGML_TYPE_F32, 256);
    send_cognitive_tensor(synthesizer, coordinator->agent_id, solution, 0.8f);
    
    // Coordinator receives and validates solution
    cognitive_tensor_packet solution_msg = {0};
    solution_msg.cognitive_type = COGNITIVE_TYPE_MEMORY;
    solution_msg.attention_weight = 0.8f;
    solution_msg.source_agent_id = synthesizer->agent_id;
    solution_msg.target_agent_id = coordinator->agent_id;
    
    process_incoming_tensor(coordinator, &solution_msg);
    
    // Display network statistics
    printf("\nNetwork communication statistics:\n");
    printf("Coordinator - Sent: %lu, Received: %lu\n", 
           coordinator->messages_sent, coordinator->messages_received);
    printf("Analyzer - Sent: %lu, Received: %lu\n", 
           analyzer->messages_sent, analyzer->messages_received);
    printf("Synthesizer - Sent: %lu, Received: %lu\n", 
           synthesizer->messages_sent, synthesizer->messages_received);
    
    printf("\nMemory statistics:\n");
    printf("Coordinator memory nodes: %zu\n", coordinator->memory->node_count);
    printf("Analyzer memory nodes: %zu\n", analyzer->memory->node_count);
    printf("Synthesizer memory nodes: %zu\n", synthesizer->memory->node_count);
    
    // Clean up
    cleanup_cognitive_agent(coordinator);
    cleanup_cognitive_agent(analyzer);
    cleanup_cognitive_agent(synthesizer);
    
    printf("\nDistributed problem solving demo completed.\n");
}

// Demo: Attention economy dynamics
void demo_attention_economy(void) {
    printf("\n=== Attention Economy Demo ===\n");
    
    cognitive_agent* agent = create_cognitive_agent("localhost:7001");
    
    printf("\nDemonstrating attention allocation dynamics...\n");
    
    // Initial state
    printf("Initial attention state:\n");
    printf("  Total: %.2f, Allocated: %.2f\n", 
           agent->attention->total_attention, agent->attention->allocated_attention);
    
    // Simulate varying cognitive demands
    printf("\nSimulating cognitive load...\n");
    
    // High memory demand
    allocate_attention(agent->attention, 0.4f, COGNITIVE_TYPE_MEMORY);
    printf("After memory allocation: %.2f/%.2f allocated\n",
           agent->attention->allocated_attention, agent->attention->total_attention);
    
    // Reasoning demand
    allocate_attention(agent->attention, 0.3f, COGNITIVE_TYPE_REASONING);
    printf("After reasoning allocation: %.2f/%.2f allocated\n",
           agent->attention->allocated_attention, agent->attention->total_attention);
    
    // Communication demand (will trigger reallocation)
    allocate_attention(agent->attention, 0.5f, COGNITIVE_TYPE_COMMUNICATION);
    printf("After communication allocation: %.2f/%.2f allocated\n",
           agent->attention->allocated_attention, agent->attention->total_attention);
    
    // Show final allocation breakdown
    printf("\nFinal attention allocation breakdown:\n");
    printf("  Memory: %.2f\n", agent->attention->memory_allocation);
    printf("  Reasoning: %.2f\n", agent->attention->reasoning_allocation);
    printf("  Communication: %.2f\n", agent->attention->communication_allocation);
    printf("  Self-modification: %.2f\n", agent->attention->self_modification_allocation);
    
    // Update performance history
    printf("\nUpdating performance history...\n");
    for (int i = 0; i < 10; i++) {
        float performance = 0.7f + 0.3f * sinf((float)i / 10.0f * 6.28f);
        update_performance_history(agent->attention, performance);
        printf("  Cycle %d: Performance %.2f\n", i, performance);
    }
    
    cleanup_cognitive_agent(agent);
    printf("\nAttention economy demo completed.\n");
}

int main(void) {
    printf("GGML Cognitive Agent Network Demo\n");
    printf("================================\n");
    
    // Run demonstrations
    demo_consciousness_exploration();
    demo_distributed_problem_solving();
    demo_attention_economy();
    
    printf("\nAll demos completed successfully!\n");
    printf("\nThis demonstrates the basic framework for distributed cognitive agents\n");
    printf("built on ggml infrastructure. In a full implementation, this would include:\n");
    printf("- Real network communication via ggml-rpc\n");
    printf("- Sophisticated reasoning engines\n");
    printf("- Grammar-based task decomposition\n");
    printf("- Self-modification capabilities\n");
    printf("- Hypergraph knowledge representation\n");
    printf("- Economic attention allocation algorithms\n");
    
    return 0;
}