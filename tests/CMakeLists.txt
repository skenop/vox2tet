add_executable(diff_tiffs tools/diff_tiffs.cpp)
target_link_libraries(diff_tiffs PRIVATE vox2tet)

add_executable(mc_only tools/mc_only.cpp)
target_link_libraries(mc_only PRIVATE vox2tet)

add_executable(diff_mesh tools/diff_mesh.cpp)
target_link_libraries(diff_mesh PRIVATE vox2tet)

add_executable(diff_xyz tools/diff_xyz.cpp)
target_link_libraries(diff_xyz PRIVATE vox2tet)

# diff_smesh — canonicalising comparison of TetGen .smesh inputs. Used
# as a debug/QA helper; not exercised by ctest.
add_executable(diff_smesh tools/diff_smesh.cpp)
# (no vox2tet linkage needed — pure self-contained text parser)

# bridge_demo — only buildable when the debug bridge is enabled.
# Exercises C++ → Python(scipy) round-trip for a synthetic EDT.
if(VOX2TET_DEBUG_BRIDGE)
    add_executable(bridge_demo tools/bridge_demo.cpp)
    target_link_libraries(bridge_demo PRIVATE vox2tet)
endif()

add_executable(test_smoke
    test_settings.cpp
    test_npy.cpp
    test_mathx.cpp
    test_label.cpp
    test_ext_volume.cpp
    test_edt.cpp
    test_remove_small.cpp
    test_critical_conn.cpp
    test_lut.cpp
    test_m3c.cpp
    test_jma_surface.cpp
    test_half_edge.cpp
    test_reseed.cpp
    test_cdt_mmg.cpp
)
target_link_libraries(test_smoke PRIVATE vox2tet GTest::gtest_main)
target_compile_definitions(test_smoke PRIVATE
    VOX2TET_DATA_DIR_TEST="${CMAKE_SOURCE_DIR}/data"
    VOX2TET_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
add_test(NAME smoke COMMAND test_smoke)
