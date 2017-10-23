cmake_minimum_required(VERSION 2.8)

#find_package( Bullet REQUIRED )

#FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
#INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

LINK_LIBRARIES(
    ${BULLET_ROOT}/lib/libBullet3Common.${DYLIB_SUFFIX} 
)

SET (OBJSOURCES
        ../Importers/ImportObjDemo/LoadMeshFromObj.cpp
        ../Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.cpp
        ../ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp
        ../Importers/ImportURDFDemo/BulletUrdfImporter.cpp
        ../Importers/ImportURDFDemo/UrdfParser.cpp
        ../Utils/b3ResourcePath.cpp
        ../ThirdPartyLibs/tinyxml/tinystr.cpp
        ../ThirdPartyLibs/tinyxml/tinyxml.cpp
        ../ThirdPartyLibs/tinyxml/tinyxmlerror.cpp
        ../ThirdPartyLibs/tinyxml/tinyxmlparser.cpp
        ../Importers/ImportURDFDemo/urdfStringSplit.cpp
        ../Importers/ImportURDFDemo/BulletUrdfImporter.cpp
        ../Importers/ImportMeshUtility/b3ImportMeshUtility.cpp
        ../Importers/ImportColladaDemo/LoadMeshFromCollada.cpp
        ../ThirdPartyLibs/stb_image/stb_image.cpp
    )

IF (WIN32)
    ADD_EXECUTABLE(App_TestHinge
		TestHingeTorque.cpp
		main.cpp
		${BULLET_PHYSICS_SOURCE_DIR}/build3/bullet.rc
        ${OBJSOURCES}
	)
ELSE()
    ADD_EXECUTABLE(App_TestHinge
		TestHingeTorque.cpp
		main.cpp
        ${OBJSOURCES}
	)
ENDIF()

TARGET_LINK_LIBRARIES(App_TestHinge ${BULLET_LIBRARIES})
TARGET_LINK_LIBRARIES(App_TestHinge ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(App_TestHinge ${HDF5_SELF_LIBRARY})
