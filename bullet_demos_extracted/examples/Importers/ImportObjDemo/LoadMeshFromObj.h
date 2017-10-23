#ifndef LOAD_MESH_FROM_OBJ_H
#define LOAD_MESH_FROM_OBJ_H


struct GLInstanceGraphicsShape;
class btCollisionShape;
class btVector3;

GLInstanceGraphicsShape* LoadMeshFromObj(const char* relativeFileName, const char* materialPrefixPath);
btCollisionShape* LoadShapesFromObj(const char* relativeFileName, const char* materialPrefixPath, const btVector3& geomScale);

#endif //LOAD_MESH_FROM_OBJ_H

