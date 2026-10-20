add_library(llama-glm-stage-plan STATIC stage-plan.cpp)
target_include_directories(llama-glm-stage-plan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(llama-glm-stage-plan PUBLIC ggml)
target_compile_features(llama-glm-stage-plan PUBLIC cxx_std_17)
set_target_properties(llama-glm-stage-plan PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(llama-glm-plan main.cpp)
target_link_libraries(llama-glm-plan PRIVATE llama-glm-stage-plan vendor::nlohmann)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS llama-glm-plan RUNTIME)
endif()
