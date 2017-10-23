# App_BasicExample is a minimal sample creating, stepping and deleting a Bullet dynamics world

cmake_minimum_required(VERSION 2.8)
#SET (BULLET_PHYSICS_SOURCE_DIR /Users/chengxuz/barrel/bullet/bullet3)

find_package(OpenGL REQUIRED)
#find_package( Bullet REQUIRED )

IF (Bullet_FOUND)
	MESSAGE("Bullet FOUND")
        #MESSAGE(${Bullet_LIBRARIES})
        #MESSAGE(${BULLET_ROOT_DIR})
ENDIF (Bullet_FOUND)


#INCLUDE_DIRECTORIES(
#${BULLET_PHYSICS_SOURCE_DIR}/src
#)

LINK_LIBRARIES(
    #/usr/local/lib/libBullet3Common.${DYLIB_SUFFIX} 
    ${BULLET_ROOT}/lib/libBullet3Common.${DYLIB_SUFFIX} 
)
#LINK_LIBRARIES(
# BulletDynamics BulletCollision LinearMath
#)
#LINK_LIBRARIES(
#    ${BULLET_LIBRARY}
#)

IF (WIN32)
	ADD_EXECUTABLE(App_BasicExample
		BasicExample.cpp
		main.cpp
		${BULLET_PHYSICS_SOURCE_DIR}/build3/bullet.rc
	)
ELSE()
	ADD_EXECUTABLE(App_BasicExample
		BasicExample.cpp
		main.cpp
	)
ENDIF()


target_link_libraries(App_BasicExample ${BULLET_LIBRARIES})

#################
# Standalone BasicExampleGui using OpenGL (but not the example browser)


INCLUDE_DIRECTORIES(
        ${BULLET_PHYSICS_SOURCE_DIR}/btgui
        .. 
        ../ThirdPartyLibs/Glew
)


SET(AppBasicExampleGui_SRCS
	BasicExample.cpp
        ${BULLET_PHYSICS_SOURCE_DIR}/build3/bullet.rc
	../StandaloneMain/main_opengl_single_example.cpp
        ../ExampleBrowser/OpenGLGuiHelper.cpp
        ../ExampleBrowser/GL_ShapeDrawer.cpp
        ../ExampleBrowser/CollisionShape2TriangleMesh.cpp
		../Utils/b3Clock.cpp
)

#this define maps StandaloneExampleCreateFunc to the right 'CreateFunc'
ADD_DEFINITIONS(-DB3_USE_STANDALONE_EXAMPLE)

#LINK_LIBRARIES(
#        BulletDynamics BulletCollision LinearMath OpenGLWindow Bullet3Common ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY}
#)
LINK_LIBRARIES(
     ${BULLET_ROOT}/lib/libBullet3Common.${DYLIB_SUFFIX} OpenGLWindow ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY}
)

LINK_LIBRARIES(
        ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY}
)

#some code to support OpenGL and Glew cross platform
IF (WIN32)
        INCLUDE_DIRECTORIES(
             ${BULLET_PHYSICS_SOURCE_DIR}/btgui/OpenGLWindow/GlewWindows
        )
        ADD_DEFINITIONS(-DGLEW_STATIC)
ELSE(WIN32)
        IF(APPLE)
                find_library(COCOA NAMES Cocoa)
                MESSAGE(${COCOA})
                link_libraries(${COCOA})

        ELSE(APPLE)
                INCLUDE_DIRECTORIES(
                        ${BULLET_PHYSICS_SOURCE_DIR}/btgui/OpenGLWindow/GlewWindows
                )
                ADD_DEFINITIONS("-DGLEW_INIT_OPENGL11_FUNCTIONS=1")
                ADD_DEFINITIONS("-DGLEW_STATIC")
                ADD_DEFINITIONS("-DGLEW_DYNAMIC_LOAD_ALL_GLX_FUNCTIONS=1")

                LINK_LIBRARIES( X11 pthread dl Xext)
        ENDIF(APPLE)
ENDIF(WIN32)


ADD_EXECUTABLE(AppBasicExampleGui
                ${AppBasicExampleGui_SRCS}
)

target_link_libraries(AppBasicExampleGui ${BULLET_LIBRARIES})
