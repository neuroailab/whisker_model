#include "LoadMeshFromObj.h"
#include"../../ThirdPartyLibs/Wavefront/tiny_obj_loader.h"
#include "../../OpenGLWindow/GLInstanceGraphicsShape.h"
#include <stdio.h> //fopen
#include "Bullet3Common/b3AlignedObjectArray.h"
#include <string>
#include <vector>
#include "Wavefront2GLInstanceGraphicsShape.h"
#include "btBulletDynamicsCommon.h"
#include "LinearMath/btVector3.h"

static float gUrdfDefaultCollisionMargin = 0.001;

GLInstanceGraphicsShape* LoadMeshFromObj(const char* relativeFileName, const char* materialPrefixPath)
{
	std::vector<tinyobj::shape_t> shapes;
	std::string err = tinyobj::LoadObj(shapes, relativeFileName, materialPrefixPath);
		
	GLInstanceGraphicsShape* gfxShape = btgCreateGraphicsShapeFromWavefrontObj(shapes);
	return gfxShape;
}

btCollisionShape* createConvexHullFromShapes(std::vector<tinyobj::shape_t>& shapes, const btVector3& geomScale)
{
	btCompoundShape* compound = new btCompoundShape();
	compound->setMargin(gUrdfDefaultCollisionMargin);

	btTransform identity;
	identity.setIdentity();

	for (int s = 0; s<(int)shapes.size(); s++)
	{
		btConvexHullShape* convexHull = new btConvexHullShape();
		convexHull->setMargin(gUrdfDefaultCollisionMargin);
		tinyobj::shape_t& shape = shapes[s];
		int faceCount = shape.mesh.indices.size();

		for (int f = 0; f<faceCount; f += 3)
		{

			btVector3 pt;
			pt.setValue(shape.mesh.positions[shape.mesh.indices[f] * 3 + 0],
				shape.mesh.positions[shape.mesh.indices[f] * 3 + 1],
				shape.mesh.positions[shape.mesh.indices[f] * 3 + 2]);
			
			convexHull->addPoint(pt*geomScale,false);

			pt.setValue(shape.mesh.positions[shape.mesh.indices[f + 1] * 3 + 0],
						shape.mesh.positions[shape.mesh.indices[f + 1] * 3 + 1],
						shape.mesh.positions[shape.mesh.indices[f + 1] * 3 + 2]);
			convexHull->addPoint(pt*geomScale, false);

			pt.setValue(shape.mesh.positions[shape.mesh.indices[f + 2] * 3 + 0],
						shape.mesh.positions[shape.mesh.indices[f + 2] * 3 + 1],
						shape.mesh.positions[shape.mesh.indices[f + 2] * 3 + 2]);
			convexHull->addPoint(pt*geomScale, false);
		}

		convexHull->recalcLocalAabb();
		convexHull->optimizeConvexHull();
		compound->addChildShape(identity,convexHull);
	}

	return compound;
}

btCollisionShape* LoadShapesFromObj(const char* relativeFileName, const char* materialPrefixPath, const btVector3& geomScale){
    std::vector<tinyobj::shape_t> shapes;
    std::string err = tinyobj::LoadObj(shapes, relativeFileName, materialPrefixPath);
    //create a convex hull for each shape, and store it in a btCompoundShape

    btCollisionShape* shape = createConvexHullFromShapes(shapes, geomScale);

    return shape;
}
