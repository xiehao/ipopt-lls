if(COMMAND cmake_minimum_required)
   cmake_minimum_required(VERSION 2.8)
endif(COMMAND cmake_minimum_required)

project(ipopt-lls)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{HOME}/usr/share/cmake/Modules/")


include_directories(
$ENV{HOME}/usr/include
./inc
)

link_directories(
$ENV{HOME}/usr/lib
$ENV{HOME}/usr/lib/Ipopt
)

file(GLOB SOURCES "./src/*.cc")
file(GLOB HEADERS "./inc/*.hh")

add_executable(ipopt-lls ${SOURCES} ${HEADERS})


set(CMAKE_CXX_FLAGS -fopenmp)
set(CMAKE_CXX_LINKFLAGS -Wl,--rpath -Wl,/usr/lib)

if(WIN32)
  target_link_libraries(ipopt-lls)
else(WIN32)
  target_link_libraries(ipopt-lls ipopt dmumps_seq dl)
endif(WIN32)
