#include "cognitive-agent.h"
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

// Generate unique agent ID
uint64_t generate_agent_id(void) {
    static uint64_t counter = 0;
    return (uint64_t)time(NULL) * 1000 + (++counter);
}

// Get current timestamp
uint64_t get_timestamp(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// Initialize hypergraph memory
hypergraph_memory* init_hypergraph_memory(struct ggml_context* ctx) {
    hypergraph_memory* mem = malloc(sizeof(hypergraph_memory));
    mem->ctx = ctx;
    mem->node_count = 0;
    mem->capacity = 1000;
    mem->nodes = malloc(sizeof(hypergraph_node*) * mem->capacity);
    mem->total_accesses = 0;
    mem->cache_hits = 0;
    return mem;
}

// Cleanup hypergraph memory
void cleanup_hypergraph_memory(hypergraph_memory* mem) {
    if (!mem) return;
    
    for (size_t i = 0; i < mem->node_count; i++) {
        free(mem->nodes[i]);
    }
    free(mem->nodes);
    free(mem);
}

// Add knowledge to hypergraph
void add_knowledge(hypergraph_memory* mem, const char* concept, 
                   float* embedding, size_t emb_size) {
    if (mem->node_count >= mem->capacity) {
        // Expand capacity
        mem->capacity *= 2;
        mem->nodes = realloc(mem->nodes, sizeof(hypergraph_node*) * mem->capacity);
    }
    
    // Create tensor for concept
    struct ggml_tensor* concept_tensor = ggml_new_tensor_1d(mem->ctx, GGML_TYPE_F32, 
                                                     strlen(concept));
    memcpy(concept_tensor->data, concept, strlen(concept));
    
    // Create embedding tensor
    struct ggml_tensor* emb_tensor = ggml_new_tensor_1d(mem->ctx, GGML_TYPE_F32, 
                                                 emb_size);
    memcpy(emb_tensor->data, embedding, emb_size * sizeof(float));
    
    // Create hypergraph node
    hypergraph_node* node = malloc(sizeof(hypergraph_node));
    node->data = concept_tensor;
    node->embedding = emb_tensor;
    node->node_type = NODE_TYPE_CONCEPT;
    node->truth_value = 0.8f;  // Initial belief
    node->confidence = 0.9f;   // High confidence
    node->creation_time = get_timestamp();
    node->last_access = node->creation_time;
    
    // Add to memory
    mem->nodes[mem->node_count++] = node;
    
    printf("Added knowledge: %s (nodes: %zu)\n", concept, mem->node_count);
}

// Find concept in hypergraph
hypergraph_node* find_concept(hypergraph_memory* mem, const char* concept) {
    mem->total_accesses++;
    
    for (size_t i = 0; i < mem->node_count; i++) {
        hypergraph_node* node = mem->nodes[i];
        if (node->node_type == NODE_TYPE_CONCEPT) {
            // Simple string comparison (in practice would use semantic matching)
            if (strncmp((char*)node->data->data, concept, strlen(concept)) == 0) {
                node->last_access = get_timestamp();
                mem->cache_hits++;
                return node;
            }
        }
    }
    return NULL;
}

// Initialize attention economy
attention_economy* init_attention_economy(void) {
    attention_economy* attention = malloc(sizeof(attention_economy));
    attention->total_attention = 1.0f;
    attention->allocated_attention = 0.0f;
    
    attention->memory_allocation = 0.0f;
    attention->reasoning_allocation = 0.0f;
    attention->communication_allocation = 0.0f;
    attention->self_modification_allocation = 0.0f;
    
    attention->min_threshold = 0.1f;
    attention->decay_rate = 0.01f;
    attention->novelty_bonus = 0.2f;
    
    attention->history_index = 0;
    attention->history_size = 0;
    
    return attention;
}

// Cleanup attention economy
void cleanup_attention_economy(attention_economy* attention) {
    if (attention) {
        free(attention);
    }
}

// Compute salience score for tensor
float compute_salience(struct ggml_tensor* tensor, attention_economy* attention) {
    // Simple salience computation (in practice would be more sophisticated)
    float base_salience = 0.5f;
    
    // Add novelty bonus
    float novelty = 0.3f; // Placeholder for novelty computation
    base_salience += novelty * attention->novelty_bonus;
    
    // Clamp to [0, 1]
    if (base_salience > 1.0f) base_salience = 1.0f;
    if (base_salience < 0.0f) base_salience = 0.0f;
    
    return base_salience;
}

// Allocate attention
void allocate_attention(attention_economy* attention, float amount, uint32_t target) {
    if (attention->allocated_attention + amount > attention->total_attention) {
        // Need to reallocate
        float excess = (attention->allocated_attention + amount) - attention->total_attention;
        // Simple strategy: reduce all allocations proportionally
        float reduction_factor = excess / attention->allocated_attention;
        attention->memory_allocation *= (1.0f - reduction_factor);
        attention->reasoning_allocation *= (1.0f - reduction_factor);
        attention->communication_allocation *= (1.0f - reduction_factor);
        attention->self_modification_allocation *= (1.0f - reduction_factor);
        
        // Recalculate allocated attention
        attention->allocated_attention = attention->memory_allocation +
                                       attention->reasoning_allocation +
                                       attention->communication_allocation +
                                       attention->self_modification_allocation;
    }
    
    // Add new allocation
    switch (target) {
        case COGNITIVE_TYPE_MEMORY:
            attention->memory_allocation += amount;
            break;
        case COGNITIVE_TYPE_REASONING:
            attention->reasoning_allocation += amount;
            break;
        case COGNITIVE_TYPE_COMMUNICATION:
            attention->communication_allocation += amount;
            break;
        default:
            attention->self_modification_allocation += amount;
            break;
    }
    
    attention->allocated_attention += amount;
    
    printf("Allocated %.2f attention to type %u (total: %.2f/%.2f)\n", 
           amount, target, attention->allocated_attention, attention->total_attention);
}

// Update performance history
void update_performance_history(attention_economy* attention, float performance) {
    attention->performance_history[attention->history_index] = performance;
    attention->history_index = (attention->history_index + 1) % 100;
    if (attention->history_size < 100) {
        attention->history_size++;
    }
}

// Initialize task orchestrator
task_orchestrator* init_task_orchestrator(void) {
    task_orchestrator* orch = malloc(sizeof(task_orchestrator));
    orch->current_goal = NULL;
    orch->context_state = NULL;
    orch->pending_tasks = NULL;
    orch->task_count = 0;
    orch->task_capacity = 0;
    return orch;
}

// Cleanup task orchestrator
void cleanup_task_orchestrator(task_orchestrator* orch) {
    if (orch) {
        if (orch->pending_tasks) {
            free(orch->pending_tasks);
        }
        free(orch);
    }
}

// Initialize reasoning engine
reasoning_engine* init_reasoning_engine(struct ggml_context* ctx) {
    reasoning_engine* reasoning = malloc(sizeof(reasoning_engine));
    reasoning->ctx = ctx;
    reasoning->current_beliefs = NULL;
    reasoning->inference_rules = NULL;
    reasoning->reasoning_accuracy = 0.75f;
    reasoning->inferences_made = 0;
    return reasoning;
}

// Cleanup reasoning engine
void cleanup_reasoning_engine(reasoning_engine* reasoning) {
    if (reasoning) {
        free(reasoning);
    }
}

// Infer cognitive type from tensor
uint32_t infer_cognitive_type(struct ggml_tensor* tensor) {
    // Simple heuristic based on tensor properties
    if (tensor->ne[0] <= 64) {
        return COGNITIVE_TYPE_ATTENTION;
    } else if (tensor->ne[0] <= 256) {
        return COGNITIVE_TYPE_MEMORY;
    } else if (tensor->ne[0] <= 1024) {
        return COGNITIVE_TYPE_REASONING;
    } else {
        return COGNITIVE_TYPE_COMMUNICATION;
    }
}

// Compute tensor similarity (simplified)
float compute_tensor_similarity(struct ggml_tensor* a, struct ggml_tensor* b) {
    if (!a || !b || a->ne[0] != b->ne[0]) {
        return 0.0f;
    }
    
    // Simple cosine similarity for float tensors
    if (a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32) {
        float* data_a = (float*)a->data;
        float* data_b = (float*)b->data;
        
        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        
        for (int i = 0; i < a->ne[0]; i++) {
            dot_product += data_a[i] * data_b[i];
            norm_a += data_a[i] * data_a[i];
            norm_b += data_b[i] * data_b[i];
        }
        
        if (norm_a == 0.0f || norm_b == 0.0f) {
            return 0.0f;
        }
        
        return dot_product / (sqrtf(norm_a) * sqrtf(norm_b));
    }
    
    return 0.0f;
}

// Create cognitive agent
cognitive_agent* create_cognitive_agent(const char* endpoint) {
    cognitive_agent* agent = malloc(sizeof(cognitive_agent));
    
    // Initialize ggml context
    struct ggml_init_params params = {
        .mem_size = 128 * 1024 * 1024,  // 128MB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    agent->ctx = ggml_init(params);
    
    // Setup backend (use CPU for simplicity in example)
    agent->backend = NULL;  // Would use ggml_backend_cpu_init() in real implementation
    
    // Initialize cognitive subsystems
    agent->memory = init_hypergraph_memory(agent->ctx);
    agent->tasks = init_task_orchestrator();
    agent->reasoning = init_reasoning_engine(agent->ctx);
    agent->attention = init_attention_economy();
    
    // Set identity
    agent->agent_id = generate_agent_id();
    strncpy(agent->endpoint, endpoint, sizeof(agent->endpoint) - 1);
    agent->endpoint[sizeof(agent->endpoint) - 1] = '\0';
    agent->attention_weight = 1.0f;
    
    // Initialize counters
    agent->messages_sent = 0;
    agent->messages_received = 0;
    agent->cycles_completed = 0;
    
    printf("Created cognitive agent %lu at %s\n", agent->agent_id, agent->endpoint);
    
    return agent;
}

// Cleanup cognitive agent
void cleanup_cognitive_agent(cognitive_agent* agent) {
    if (!agent) return;
    
    cleanup_hypergraph_memory(agent->memory);
    cleanup_task_orchestrator(agent->tasks);
    cleanup_reasoning_engine(agent->reasoning);
    cleanup_attention_economy(agent->attention);
    
    if (agent->ctx) {
        ggml_free(agent->ctx);
    }
    
    printf("Cleaned up cognitive agent %lu\n", agent->agent_id);
    free(agent);
}

// Send cognitive tensor (simplified - no actual network communication)
void send_cognitive_tensor(cognitive_agent* sender, uint64_t target_agent_id,
                          struct ggml_tensor* tensor, float attention_weight) {
    
    cognitive_tensor_packet packet = {0};
    
    // In real implementation, would serialize tensor using ggml-rpc
    packet.attention_weight = attention_weight;
    packet.salience_score = compute_salience(tensor, sender->attention);
    packet.cognitive_type = infer_cognitive_type(tensor);
    packet.source_agent_id = sender->agent_id;
    packet.target_agent_id = target_agent_id;
    packet.recursion_depth = 0;
    packet.timestamp = get_timestamp();
    
    strncpy(packet.meta_context, "cognitive_exchange", sizeof(packet.meta_context) - 1);
    
    sender->messages_sent++;
    
    printf("Agent %lu sent cognitive tensor (type %u, attention %.2f, salience %.2f) to agent %lu\n",
           sender->agent_id, packet.cognitive_type, packet.attention_weight, 
           packet.salience_score, target_agent_id);
}

// Process incoming tensor (simplified)
void process_incoming_tensor(cognitive_agent* receiver, 
                           cognitive_tensor_packet* msg) {
    
    receiver->messages_received++;
    
    printf("Agent %lu received cognitive tensor from agent %lu (type %u, attention %.2f)\n",
           receiver->agent_id, msg->source_agent_id, msg->cognitive_type, msg->attention_weight);
    
    // Check attention allocation
    if (msg->attention_weight < receiver->attention->min_threshold) {
        printf("  Insufficient attention weight, deferring message\n");
        return;
    }
    
    // Allocate attention for processing
    allocate_attention(receiver->attention, msg->attention_weight, msg->cognitive_type);
    
    // Process based on cognitive type
    switch (msg->cognitive_type) {
        case COGNITIVE_TYPE_MEMORY:
            printf("  Processing memory operation\n");
            // In real implementation: integrate_memory(receiver->memory, tensor, msg);
            break;
            
        case COGNITIVE_TYPE_TASK:
            printf("  Processing task request\n");
            // In real implementation: process_task_request(receiver->tasks, tensor, msg);
            break;
            
        case COGNITIVE_TYPE_REASONING:
            printf("  Processing reasoning request\n");
            receiver->reasoning->inferences_made++;
            break;
            
        case COGNITIVE_TYPE_ATTENTION:
            printf("  Processing attention update\n");
            break;
            
        default:
            printf("  Processing unknown cognitive type\n");
            break;
    }
}