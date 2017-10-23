#include "TestHingeTorque.h"
#include "../ExtendedTutorials/RigidBodyFromObj.h"
#include <boost/program_options.hpp>

#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>

#include "btBulletDynamicsCommon.h"
#include "LinearMath/btVector3.h"
#include "LinearMath/btAlignedObjectArray.h" 
#include "../Importers/ImportObjDemo/LoadMeshFromObj.h"
#include "../OpenGLWindow/GLInstanceGraphicsShape.h"
#include "BulletCollision/NarrowPhaseCollision/btRaycastCallback.h"


namespace po = boost::program_options;
using namespace std;

#include "../CommonInterfaces/CommonRigidBodyBase.h"
#include "../CommonInterfaces/CommonParameterInterface.h"


#include "../Importers/ImportURDFDemo/BulletUrdfImporter.h"

#include "H5Cpp.h"

short collisionFilterGroup = short(btBroadphaseProxy::CharacterFilter);
short collisionFilterMask = short(btBroadphaseProxy::AllFilter ^ (btBroadphaseProxy::CharacterFilter));

vector<float> x_pos_base    = {-0.4};
vector<float> y_pos_base    = {7};
vector<float> z_pos_base    = {0};
vector<int> const_numLinks  = {15};
vector<float> qua_a_list    = {-1};
vector<float> yaw_y_base    = {0.5};
vector<float> pitch_x_base  = {0};
vector<float> roll_z_base   = {0};
vector<int> inter_spring = {5};
vector<int> every_spring = {1};

const float SMALL_NUM = 1e-6;

float x_len_link    = 0.53;
float y_len_link    = 2.08;
float z_len_link    = 0.3;
float radius        = 1;
float time_leap     = 1.0/240.0;
vector<string> whisker_config_name = {"test.cfg"};

float camera_dist       = 45;
float camera_yaw        = 21;
float camera_pitch      = 270;

float time_limit    = 60.0;
float initial_str   = 10000;
float max_str   = 10000;
float initial_stime = 1.0/8;
int initial_poi     = 14;
int flag_time       = 0;
float limit_softness = 0.9;
float limit_bias    = 0.3;
float limit_relax   = 1;
float limit_low     = -2;
float limit_up      = 0;
float velo_ban_limit    = 1;
float angl_ban_limit    = 1;
float force_limit       = 1;
float torque_limit      = 1;
float dispos_limit      = 20;

int test_mode           = 0;
int force_mode          = 0;

float percision         = 0.001;

int add_objs            = 0;
vector<string> obj_filename = {"/Users/chengxuz/barrel/bullet/bullet3/data/teddy.obj"};
vector<float> obj_scaling_list = {0.5,0.5,0.5,1};
vector<float> obj_pos_list = {-20,20,-30,0};
vector<float> obj_orn_list = {0,0,0,1};
vector<float> obj_speed_list = {0,-5,0};
vector<float> obj_mass_list = {100};
vector<float> control_len = {-1};
vector<float> offset_center_pos = {0,0,0};
int reset_pos           = 1;
int reset_speed         = 0;
int avoid_coll          = 0;
float avoid_coll_z_off  = 0;
float avoid_coll_x_off  = 5;
int get_normal          = 1;
int normal_im_h         = 256;
int normal_im_w         = 256;
float normal_unit_len   = 0.2;

int do_save             = 0;
H5std_string FILE_NAME( "Select.h5" );
int num_unit_to_save    = 3;
int sample_rate         = 24;
string group_name       = "";

struct TestHingeTorque : public CommonRigidBodyBase{
    bool m_once;
    float pass_time;
    float curr_velo;
    float curr_angl;
    float curr_force;
    float curr_torque;
    float curr_dispos;
    float loss_ret; // Final return value, the loss function for hyperopt optimization
    float min_dis;

    btAlignedObjectArray< btAlignedObjectArray< btRigidBody* > > m_allbones_big_list; // store all units
    btAlignedObjectArray< btAlignedObjectArray<btJointFeedback*> > m_jointFeedback_big_list; // store all force, torque feedback reader for springs
    btAlignedObjectArray< btAlignedObjectArray< btVector3 > > m_allcentpos_big_list; // store all center position for units
    btAlignedObjectArray< btRigidBody* > m_allbaseballs; // store all base balls
    btAlignedObjectArray< btRigidBody* > m_allobjs; // store all objs
    btAlignedObjectArray< btTransform > m_objstartTrans; // store all start transform for all objs
    btAlignedObjectArray< btVector3 > m_objcenter; // store center position calculated from bounding box for all objs, before start trans
    btAlignedObjectArray< btVector3 > m_objboundingbox; // store bounding box for all objs, before start trans
    vector< vector<int> > m_spring_strtindx;

    vector< vector<float> > m_base_spring_stiffness;
    vector< vector<float> > m_spring_stfperunit_list;
    vector< float > m_basic_str;
    vector< float > m_base_ball_base_spring_stf;
    vector< float > m_spring_stfperunit;
    vector< float > m_linear_damp;
    vector< float > m_ang_damp;

    vector< vector< vector< vector<float> > > > all_force_data;
    vector< vector< vector< vector<float> > > > all_torque_data;
    vector< vector< vector< vector<unsigned char> > > > all_normals;

    btVector3 base_ball_location;
    btTransform base_ball_trans;
    btVector3 cent_of_whisker_array;
    btVector3 orig_cent_of_whisker_array;

	TestHingeTorque(struct GUIHelperInterface* helper);
	virtual ~ TestHingeTorque();
	virtual void initPhysics();

	virtual void stepSimulation(float deltaTime);
    void addQuaUnits(float qua_a, float qua_b, float qua_c, int num_units, 
            btTransform base_transform);
    btJointFeedback* addFeedbackForSpring(btGeneric6DofSpring2Constraint* con);
    void loadParametersEveryWhisker(string curr_file_name, po::options_description desc_each);
    btRigidBody* addObjasRigidBody(string fileName, float scaling[4], float orn[4], float pos[4] , float mass_want, float control_len_now, int indx_obj );
    void save_all_data(string );
    void cal_cent_whisker();
    vector< vector< vector<unsigned char> > > get_normal_picture(btVector3 from_p, btVector3 to_p, int image_h, int image_w, btVector3 unit_x, btVector3 unit_y, float control_len_now);
    void set_initial_speed_all(btVector3 initial_speed);
    void set_initial_speed_baseballs(btVector3 initial_speed);
    void set_initial_speed_units(btVector3 initial_speed);
    void set_speed_objs(btVector3 );
	
	virtual void resetCamera(){
        //float dist = 5;
        float dist = camera_dist;
        //float dist = 15;
        float pitch = camera_pitch;
        float yaw = camera_yaw;
        //float targetPos[3]={-1.34,3.4,-0.44};
        float targetPos[3]={-1.34,5.4,-0.44};
        m_guiHelper->resetCamera(dist,pitch,yaw,targetPos[0],targetPos[1],targetPos[2]);
	}
};

void TestHingeTorque::set_speed_objs(btVector3 speed_want){
    for (int i=0;i < m_allobjs.size();i++){
        btRigidBody* curr_body = m_allobjs[i];
        curr_body->setLinearVelocity(speed_want);
        curr_body->setAngularVelocity(btVector3(0,0,0));
    }
}

void TestHingeTorque::set_initial_speed_all(btVector3 initial_speed){
    set_initial_speed_baseballs(initial_speed);
    set_initial_speed_units(initial_speed);
}

void TestHingeTorque::set_initial_speed_baseballs(btVector3 initial_speed){
    for (int indx_baseball=0;indx_baseball < m_allbaseballs.size();indx_baseball++){
        btRigidBody* curr_body = m_allbaseballs[indx_baseball];
        curr_body->setLinearVelocity(initial_speed);
        curr_body->setAngularVelocity(btVector3(0,0,0));
    }
}

void TestHingeTorque::set_initial_speed_units(btVector3 initial_speed){
    for (int i=0;i < m_allbones_big_list.size();i++){
        btAlignedObjectArray< btRigidBody* > curr_unit_list = m_allbones_big_list[i];
        for (int j=0;j < curr_unit_list.size();j++){
            btRigidBody* curr_body = curr_unit_list[j];
            curr_body->setLinearVelocity(initial_speed);
            curr_body->setAngularVelocity(btVector3(0,0,0));
        }
    }
}

void TestHingeTorque::cal_cent_whisker(){
    btAlignedObjectArray< btVector3 > curr_centposes;
    cent_of_whisker_array   = btVector3(0,0,0);
    int num_units_all = 0;
    for (int i=0;i<m_allcentpos_big_list.size();i++){
        curr_centposes      = m_allcentpos_big_list[i];
        btVector3 curr_centpos;
        for (int j=0;j<curr_centposes.size();j++){
            curr_centpos    = curr_centposes[j];
            cent_of_whisker_array += curr_centpos;
            num_units_all++;
        }
    }
    cent_of_whisker_array /= num_units_all;
    orig_cent_of_whisker_array = btVector3(cent_of_whisker_array);

    for (int i=0;i<3;i++)
        cent_of_whisker_array[i] += offset_center_pos[i];

    // Test code
    // cout << cent_of_whisker_array[0] << " " << cent_of_whisker_array[1] << " " << cent_of_whisker_array[2] << endl;
}

void save_oned_array(H5::H5File* file, vector<float> array_to_save, string dataset_name, H5::DSetCreatPropList plist){
    int size_array = array_to_save.size();
    float* all_data_in_array = new float[size_array];
    for (int i=0;i<size_array;i++)
        all_data_in_array[i] = array_to_save[i];

    hsize_t fdim[] = {(hsize_t) size_array};

    H5::DataSet* dataset = new H5::DataSet(file->createDataSet(
        dataset_name, H5::PredType::NATIVE_FLOAT, H5::DataSpace(1, fdim), plist));

    dataset->write( all_data_in_array, H5::PredType::NATIVE_FLOAT);

    delete dataset;
    delete [] all_data_in_array;
}

void save_fourd_array(H5::H5File* file, vector< vector< vector< vector<unsigned char> > > > array_to_save, string dataset_name, H5::DSetCreatPropList plist){

    hsize_t fdim[] = {array_to_save.size(), array_to_save[0].size(), array_to_save[0][0].size(), array_to_save[0][0][0].size()}; // dim sizes of ds (on disk)
    H5::DataSpace fspace( 4, fdim );
    unsigned char* all_data_in_array;
    int whole_len = fdim[0]*fdim[1]*fdim[2]*fdim[3];
    all_data_in_array = new unsigned char[whole_len];

    H5::DataSet* dataset = new H5::DataSet(file->createDataSet(
        dataset_name, H5::PredType::NATIVE_UCHAR, fspace, plist));

    int now_indx = 0;
    for (int i=0;i<fdim[0];i++)
        for (int j=0;j<fdim[1];j++)
            for (int k=0;k<fdim[2];k++)
                for (int l=0;l<fdim[3];l++){
                    all_data_in_array[now_indx] = array_to_save[i][j][k][l];
                    now_indx++;
                }

    dataset->write( all_data_in_array, H5::PredType::NATIVE_UCHAR);
    delete dataset;
    delete [] all_data_in_array;
}

void save_fourd_array(H5::H5File* file, vector< vector< vector< vector<float> > > > array_to_save, string dataset_name, H5::DSetCreatPropList plist){

    hsize_t fdim[] = {array_to_save.size(), array_to_save[0].size(), array_to_save[0][0].size(), array_to_save[0][0][0].size()}; // dim sizes of ds (on disk)
    H5::DataSpace fspace( 4, fdim );
    float* all_data_in_array;
    int whole_len = fdim[0]*fdim[1]*fdim[2]*fdim[3];
    all_data_in_array = new float[whole_len];

    H5::DataSet* dataset = new H5::DataSet(file->createDataSet(
        dataset_name, H5::PredType::NATIVE_FLOAT, fspace, plist));

    int now_indx = 0;
    for (int i=0;i<fdim[0];i++)
        for (int j=0;j<fdim[1];j++)
            for (int k=0;k<fdim[2];k++)
                for (int l=0;l<fdim[3];l++){
                    all_data_in_array[now_indx] = array_to_save[i][j][k][l];
                    now_indx++;
                }

    dataset->write( all_data_in_array, H5::PredType::NATIVE_FLOAT);
    delete dataset;
    delete [] all_data_in_array;
}

void TestHingeTorque::save_all_data(string group_name = ""){

    //H5::H5File* file = new H5::H5File( FILE_NAME, H5F_ACC_TRUNC );
    H5::H5File* file = 0;
	try {
		file = new H5::H5File(FILE_NAME, H5F_ACC_RDWR);
	} catch(H5::FileIException &file_exists_err) {
		file = new H5::H5File(FILE_NAME, H5F_ACC_TRUNC);
	}

    float fillvalue = 0;  
    H5::DSetCreatPropList plist;
    plist.setFillValue(H5::PredType::NATIVE_FLOAT, &fillvalue);

    string group_prefix = group_name + "/";
    H5::Group* group = 0;

    if (group_name.length()>0)
        group = new H5::Group( file->createGroup( group_name ));
    else
        group_prefix = "";

    unsigned char fillvalue_uc = 0;  
    H5::DSetCreatPropList plist_uc;
    plist_uc.setFillValue(H5::PredType::NATIVE_UCHAR, &fillvalue_uc);

    save_fourd_array(file, all_force_data, group_prefix + "Data_force", plist);
    save_fourd_array(file, all_torque_data, group_prefix + "Data_torque", plist);
    save_fourd_array(file, all_normals, group_prefix + "Data_normal", plist_uc);

    save_oned_array(file, obj_scaling_list, group_prefix + "scale", plist);
    save_oned_array(file, obj_pos_list, group_prefix + "position", plist);
    save_oned_array(file, obj_orn_list, group_prefix + "orn", plist);
    save_oned_array(file, obj_speed_list, group_prefix + "speed", plist);

    if (group) delete group;
    delete file;

}

TestHingeTorque::TestHingeTorque(struct GUIHelperInterface* helper)
:CommonRigidBodyBase(helper),
m_once(true){
}
TestHingeTorque::~ TestHingeTorque(){
    /*
	for (int i=0;i<m_jointFeedback.size();i++){
		delete m_jointFeedback[i];
	}
    */
}

vector< vector< vector<unsigned char> > > TestHingeTorque::get_normal_picture(btVector3 from_p, btVector3 to_p, int image_h, int image_w, btVector3 unit_x, btVector3 unit_y, float control_len_now){
    vector< vector< vector<unsigned char> > > normal_picture;
    normal_picture.clear();

    if (control_len_now>0) {
        btVector3 diff_vec = from_p - to_p;
        diff_vec = diff_vec/diff_vec.norm()*control_len_now*3;
        from_p = to_p + diff_vec;
    }

    to_p = 2*to_p - from_p;

    int sta_x = -image_h/2, sta_y = -image_w/2;

    for (int indx_x=sta_x;indx_x<sta_x+image_h;indx_x++){
        vector< vector<unsigned char> > curr_row;
        curr_row.clear();

        for (int indx_y=sta_y;indx_y<sta_y+image_w;indx_y++){
            vector<unsigned char> curr_pos;
            curr_pos.clear();

            btVector3 new_from = from_p + indx_x*unit_x + indx_y*unit_y;
            btVector3 new_to = to_p + indx_x*unit_x + indx_y*unit_y;
            
            btCollisionWorld::ClosestRayResultCallback	closestResults(new_from,new_to);
            closestResults.m_collisionFilterGroup   = collisionFilterGroup;
            closestResults.m_collisionFilterMask    = collisionFilterMask;
            
            m_dynamicsWorld->rayTest(new_from,new_to,closestResults);

            btVector3 now_normal(0,0,0);

            if (closestResults.hasHit()){
                now_normal = closestResults.m_hitNormalWorld;
            }

            now_normal.normalize();
            now_normal = now_normal*0.5 + btVector3(0.5, 0.5, 0.5);
            now_normal = 255*now_normal;

            for (int indx_tmp=0;indx_tmp<3;indx_tmp++)
                curr_pos.push_back((unsigned char) now_normal[indx_tmp]);

            curr_row.push_back(curr_pos);
        }
        normal_picture.push_back(curr_row);
    }

    return normal_picture;
}

void TestHingeTorque::stepSimulation(float deltaTime){
    
    m_dynamicsWorld->stepSimulation(time_leap,0);
    pass_time   = pass_time + time_leap;
    if (flag_time==1){
        if (pass_time > time_limit){
            save_all_data(group_name);
            exit(0);
        }
    }

    // Raytest tmp test
    if (false){
		btVector3 blue(0,0,1);
        btVector3 from(cent_of_whisker_array);
        btVector3 to(m_allobjs[0]->getCenterOfMassPosition());
        m_dynamicsWorld->getDebugDrawer()->drawLine(from,to,btVector4(0,0,1,1));

        btCollisionWorld::ClosestRayResultCallback	closestResults(from,to);
        closestResults.m_collisionFilterGroup   = collisionFilterGroup;
        closestResults.m_collisionFilterMask    = collisionFilterMask;
        
        m_dynamicsWorld->rayTest(from,to,closestResults);

        if (closestResults.hasHit()){
            
            btVector3 p = from.lerp(to,closestResults.m_closestHitFraction);
            m_dynamicsWorld->getDebugDrawer()->drawSphere(p,0.1,blue);
            m_dynamicsWorld->getDebugDrawer()->drawLine(p,p+closestResults.m_hitNormalWorld,blue);

        }
    }

    curr_velo   = 0; //Linear speed
    curr_angl   = 0; //angle speed
    curr_force  = 0; //force applied by springs 
    curr_torque = 0; // torque applied by springs
    curr_dispos = 0; // position from balance position

    static int count = 0;
    int all_size_for_big_list   = 0;
    int all_size_for_fb         = 0;
    int num_all_saved_springs   = 0;

    vector< vector< vector<float> > > m_curr_force_data;
    vector< vector< vector<float> > > m_curr_torque_data;

    for (int big_list_indx=0;big_list_indx < const_numLinks.size(); big_list_indx++) {
        btAlignedObjectArray< btRigidBody* > m_allbones;
        btAlignedObjectArray< btJointFeedback* > m_jointFeedback;
        btAlignedObjectArray< btVector3 > m_allcentpos;
        vector< int > spring_strtindx;
        //btAlignedObjectArray< btVector3 > curr_force_data;
        //btAlignedObjectArray< btVector3 > curr_torque_data;
        vector< vector<float> > curr_force_data;curr_force_data.clear();
        vector< vector<float> > curr_torque_data;curr_torque_data.clear();

        for (int i=0;i<num_unit_to_save;i++)
            curr_force_data.push_back(vector<float>({0, 0, 0}));
        for (int i=0;i<num_unit_to_save;i++)
            curr_torque_data.push_back(vector<float>({0, 0, 0}));

        m_allbones      = m_allbones_big_list[big_list_indx];
        m_jointFeedback = m_jointFeedback_big_list[big_list_indx];
        m_allcentpos    = m_allcentpos_big_list[big_list_indx];
        spring_strtindx = m_spring_strtindx[big_list_indx];

        int all_size    = m_allbones.size();

        if (flag_time==2){
            for (int i=0;i<all_size;i++){
                float curr_speed_norm = m_allbones[i]->getLinearVelocity().norm();
                if ((pass_time >= initial_stime)){
                    loss_ret += curr_speed_norm*time_leap;
                }
                curr_velo   += curr_speed_norm;
                curr_angl   += m_allbones[i]->getAngularVelocity().norm();

                btVector3 curr_pos = m_allbones[i]->getCenterOfMassPosition();
                curr_pos = curr_pos - m_allcentpos[i];
                curr_dispos += curr_pos.norm();
            }

            for (int i=0;i<m_jointFeedback.size();i++){
                curr_force  += m_jointFeedback[i]->m_appliedForceBodyA.norm();
                curr_force  += m_jointFeedback[i]->m_appliedForceBodyB.norm();

                curr_torque += m_jointFeedback[i]->m_appliedTorqueBodyA.norm();
                curr_torque += m_jointFeedback[i]->m_appliedTorqueBodyB.norm();
            }
        }

        if ((count % sample_rate==0) && (do_save!=0)){
            for (int i=0;i<spring_strtindx.size();i++){
                if (spring_strtindx[i]<num_unit_to_save){
                    num_all_saved_springs++;
                    btVector3 tmp_vec = m_jointFeedback[i]->m_appliedForceBodyA;
                    for (int j=0;j<3;j++)
                        curr_force_data[spring_strtindx[i]][j] += tmp_vec[j];

                    tmp_vec     = m_jointFeedback[i]->m_appliedTorqueBodyA;
                    for (int j=0;j<3;j++)
                        curr_torque_data[spring_strtindx[i]][j] += tmp_vec[j];
                }
            }
            //cout << curr_force_data.size() << " " << curr_torque_data.size() << endl;
        }

        all_size_for_big_list   += all_size;
        all_size_for_fb         += m_jointFeedback.size();

        if ((pass_time < initial_stime)){

            btVector3 base_loc  = m_allbones[0]->getCenterOfMassPosition();
            btVector3 end_loc   = m_allbones[m_allbones.size()-1]->getCenterOfMassPosition();
            btVector3 direc_f   = base_loc - end_loc;
            if (direc_f.norm() < min_dis) min_dis = direc_f.norm();

            //cout << force_mode << endl;
            if (force_mode==0){
                direc_f.normalize();
                m_allbones[m_allbones.size()-1]->applyForce(initial_str*direc_f, btVector3(0,0,0));
            } else if (force_mode==1){
                direc_f = base_ball_location - base_ball_trans( btVector3(0,0,-1));
                direc_f.normalize();
                float force_needed = initial_str * pass_time;
                if (force_needed > max_str ) force_needed = max_str;
                m_allbones[m_allbones.size()-1]->applyForce(force_needed*direc_f, btVector3(0,0,0));
            } else if (force_mode==2){
                direc_f = base_ball_location - base_ball_trans( btVector3(1,0,0));
                direc_f.normalize();
                float force_needed = initial_str * pass_time;
                if (force_needed > max_str ) force_needed = max_str;
                m_allbones[m_allbones.size()-1]->applyForce(force_needed*direc_f, btVector3(0,0,0));
            } else if (force_mode==3){
                direc_f = base_ball_location - base_ball_trans( btVector3(0,0,-1));
                direc_f.normalize();
                float force_needed = initial_str * pass_time;
                if (force_needed > max_str ) force_needed = max_str;
                m_allbones[m_allbones.size()/2]->applyForce(force_needed*direc_f, btVector3(0,0,0));
            }

        }
        if ((count % sample_rate==0) && (do_save!=0)){
            m_curr_force_data.push_back(curr_force_data);
            m_curr_torque_data.push_back(curr_torque_data);
            //cout << num_all_saved_springs << endl;
        }
    }

    if ((count % sample_rate==0) && (do_save!=0)){
        all_force_data.push_back(m_curr_force_data);
        all_torque_data.push_back(m_curr_torque_data);
        //cout << m_curr_force_data[0][1][0] << " " << m_curr_force_data[0][1][1] << " " << m_curr_force_data[0][1][2] << endl;
    }

    count++;
    curr_velo   /= all_size_for_big_list;
    curr_angl   /= all_size_for_big_list;
    curr_force  /= all_size_for_fb;
    curr_torque /= all_size_for_fb;

    //set_initial_speed_baseballs(btVector3(-obj_speed_list[0],-obj_speed_list[1],-obj_speed_list[2]));
    //cout << "Now state:" << curr_velo << " " << curr_angl << " " << curr_force << " " << curr_torque << " " << curr_dispos << endl;
    set_speed_objs(btVector3(obj_speed_list[0],obj_speed_list[1],obj_speed_list[2]));

    if ((flag_time==2) && (pass_time > initial_stime)){
        if (((curr_velo < velo_ban_limit) && (curr_angl < angl_ban_limit) && (curr_force < force_limit) && (curr_torque < torque_limit) && (curr_dispos < dispos_limit)) || (pass_time > time_limit)){
            cout << "Now state:" << curr_velo << " " << curr_angl << " " << curr_force << " " << curr_torque << " " << curr_dispos << endl;
            cout << "Current distance: " << loss_ret << endl;
            cout << "Mini distance: " << min_dis << endl;
            if (pass_time <= time_limit) 
                cout << "Passed time: " << pass_time << endl;
            else
                cout << "Passed time: " << pass_time+1000 << endl;
            exit(0);
        }
    }

}


float getValueQua(float qua_a, float qua_b, float qua_c, float x_now){
    float y_now = qua_a*pow(x_now, 2) + qua_b*x_now + qua_c;
    return y_now;
}

float getNorm(float x_now, float y_now){
    float norm_now = sqrt(pow(x_now, 2) + pow(y_now, 2));
    return norm_now;
}

float getDistanceQua(float qua_a, float qua_b, float qua_c, float cir_x0, float x_now){
    float cir_y0 = getValueQua(qua_a, qua_b, qua_c, cir_x0);
    float y_now = getValueQua(qua_a, qua_b, qua_c, x_now);
    float distance_now = getNorm(cir_x0 - x_now, cir_y0 - y_now);

    return distance_now;
}

float findInterQuaCirleBin(float qua_a, float qua_b, float qua_c, float cir_x0, float cir_r0, 
        float percision, float direction = 1){
    float curr_distance = 20*direction;
    while (getDistanceQua(qua_a, qua_b, qua_c, cir_x0, cir_x0 + curr_distance) < cir_r0) {
        curr_distance = curr_distance*2;
    }

    float max_dis = curr_distance;
    float min_dis = 0;

    float final_val = 0;

    while (true){
        float now_dis_s = getDistanceQua(qua_a, qua_b, qua_c, cir_x0, cir_x0 + min_dis);
        if (abs(now_dis_s - cir_r0) < percision){
            final_val = cir_x0 + min_dis;
            break;
        }
        float now_dis_b = getDistanceQua(qua_a, qua_b, qua_c, cir_x0, cir_x0 + max_dis);
        if (abs(now_dis_b - cir_r0) < percision){
            final_val = cir_x0 + max_dis;
            break;
        }

        curr_distance = (max_dis + min_dis)/2;
        if (getDistanceQua(qua_a, qua_b, qua_c, cir_x0, cir_x0 + curr_distance) < cir_r0) {
            min_dis = curr_distance;
        } else {
            max_dis = curr_distance;
        }
    }

    return final_val;
}

btVector3 findLineInter(btVector3 pre_1, btVector3 next_1, btVector3 pre_2, btVector3 next_2){
    float x_1 = pre_1[2], y_1 = pre_1[1], x_2 = next_1[2], y_2 = next_1[1];
    float x_p_1 = pre_2[2], y_p_1 = pre_2[1], x_p_2 = next_2[2], y_p_2 = next_2[1]; 

    float a_11 = x_2 - x_1, a_12 = y_1 - y_2;
    float a_21 = x_p_2 - x_p_1, a_22 = y_p_1 - y_p_2;

    //cout << a_11 << " " << a_12 << " " << a_21 << " " << a_22 << endl;

    float abs_mat = a_11*a_22 - a_21*a_12;

    if (abs(abs_mat) < SMALL_NUM){
        return (pre_2 + next_1)/2;
    }

    float b_1 = x_2*y_1 - x_1*y_2, b_2 = x_p_2*y_p_1 - x_p_1*y_p_2;

    //cout << b_1 << " " << b_2 << endl;

    float inv_abs = (1/abs_mat);

    float new_y = inv_abs*(a_22*b_1 - a_12*b_2), new_z = inv_abs*(a_11*b_2 - a_21*b_1);
    return btVector3(0, new_y, new_z);
}


btJointFeedback* TestHingeTorque::addFeedbackForSpring(btGeneric6DofSpring2Constraint* con){
    btJointFeedback* fb = new btJointFeedback();
    con->setJointFeedback(fb);
    return fb;
}

void TestHingeTorque::addQuaUnits(float qua_a, float qua_b, float qua_c, int indx_units, 
        btTransform base_transform = btTransform(btQuaternion::getIdentity(),btVector3(0, 0, 0))){
    float pre_z     = 0;
    float pre_y     = getValueQua(qua_a, qua_b, qua_c, pre_z);
    float pre_deg   = 0;
    int num_units   = const_numLinks[indx_units];

    vector<float> base_spring_stiffness = m_base_spring_stiffness[indx_units];
    vector<float> spring_stfperunit_list = m_spring_stfperunit_list[indx_units];
    float basic_str     = m_basic_str[indx_units];
    float base_ball_base_spring_stf = m_base_ball_base_spring_stf[indx_units];
    float spring_stfperunit     = m_spring_stfperunit[indx_units];
    float linear_damp   = m_linear_damp[indx_units];
    float ang_damp      = m_ang_damp[indx_units];

    btRigidBody* pre_unit = 0;
        
    float baseMass = 0.f;
    //float baseMass = obj_mass_list[0];
    float linkMass = 1.f;

    btAlignedObjectArray< btRigidBody* > m_allbones;
    btAlignedObjectArray< btVector3 > m_allpre;
    btAlignedObjectArray< btVector3 > m_allnext;
    btAlignedObjectArray<btJointFeedback*> m_jointFeedback;
    btAlignedObjectArray< btVector3 > m_allcentpos;
    vector<int> spring_strtindx;
    vector<float> m_alldeg;

    m_allbones.clear();
    m_allpre.clear();
    m_allnext.clear();
    m_alldeg.clear();
    m_allcentpos.clear();

    btVector3 linkHalfExtents(x_len_link, y_len_link, z_len_link);
    btBoxShape* baseBox = new btBoxShape(linkHalfExtents);
    btSphereShape* linkSphere = new btSphereShape(radius);

    for (int indx_unit=0;indx_unit < num_units;indx_unit++){
        float next_z = findInterQuaCirleBin(qua_a, qua_b, qua_c, pre_z, 2*y_len_link, percision, 1);
        float next_y = getValueQua(qua_a, qua_b, qua_c, next_z);
        float base_y_tmp = (pre_y + next_y)/2;
        float base_z_tmp = (pre_z + next_z)/2;

        float deg_away = atan2(next_z - pre_z, next_y - pre_y);
        btQuaternion test_rotation = btQuaternion( 0, deg_away, 0);
        btVector3 basePosition = btVector3( 0, base_y_tmp, base_z_tmp);
        btTransform baseWorldTrans(test_rotation, basePosition);
        baseWorldTrans  = base_transform*baseWorldTrans;

        btRigidBody* base = createRigidBody(linkMass,baseWorldTrans,baseBox);

        m_dynamicsWorld->removeRigidBody(base);
        base->setDamping(linear_damp,ang_damp);
        m_dynamicsWorld->addRigidBody(base,collisionFilterGroup,collisionFilterMask);
        //m_dynamicsWorld->addRigidBody(base);

        //base->setLinearVelocity(btVector3(-obj_speed_list[0],-obj_speed_list[1],-obj_speed_list[2]));
        //base->setAngularVelocity(btVector3(0,0,0));

        if (pre_unit){
            for (float x_land_pos=-x_len_link;x_land_pos < x_len_link*1.1;x_land_pos+=x_len_link){

                btTransform pivotInA(btQuaternion::getIdentity(),btVector3(x_land_pos, y_len_link, 0));						//par body's COM to cur body's COM offset
                btTransform pivotInB(btQuaternion::getIdentity(),btVector3(x_land_pos, -y_len_link, 0));							//cur body's COM to cur body's PIV offset
                btGeneric6DofSpring2Constraint* fixed = new btGeneric6DofSpring2Constraint(*pre_unit, *base, pivotInA, pivotInB);

                for (int indx_tmp=3;indx_tmp<6;indx_tmp++){
                    fixed->enableSpring(indx_tmp, true);
                    float tmp_stiff = basic_str + spring_stfperunit*(num_units - indx_unit -1);
                    tmp_stiff = tmp_stiff/3;
                    if (tmp_stiff<100) tmp_stiff = 100;
                    fixed->setStiffness(indx_tmp, tmp_stiff);
                }

                fixed->setEquilibriumPoint(3, -(deg_away - pre_deg));

                for (int indx_tmp=0;indx_tmp<3;indx_tmp++){
                    fixed->enableSpring(indx_tmp, true);
                    float tmp_stiff = basic_str + spring_stfperunit*(num_units - indx_unit -1);
                    if (tmp_stiff<100) tmp_stiff = 100;
                    fixed->setStiffness(indx_tmp, tmp_stiff);
                }

                m_jointFeedback.push_back(addFeedbackForSpring(fixed));
                spring_strtindx.push_back(indx_unit);

                m_dynamicsWorld->addConstraint(fixed,true);
            }

            /*
            btVector3 axisInA(1,0,0);
            btVector3 axisInB(1,0,0);
            btVector3 pivotInA_h(0, y_len_link,0);
            btVector3 pivotInB_h(0,-y_len_link,0);
            bool useReferenceA = true;
            btHingeConstraint* hinge = new btHingeConstraint(*pre_unit,*base,
                pivotInA_h,pivotInB_h,
                axisInA,axisInB,useReferenceA);
            m_dynamicsWorld->addConstraint(hinge,true);
            */
        } else {
            btVector3 basePosition_ball = btVector3(0, -radius-y_len_link, 0);
            btTransform baseWorldTrans_ball;
            baseWorldTrans_ball.setIdentity();
            baseWorldTrans_ball.setOrigin(basePosition_ball);
            baseWorldTrans_ball     = baseWorldTrans*baseWorldTrans_ball;

            btRigidBody* base_ball = createRigidBody(baseMass,baseWorldTrans_ball,linkSphere);
            m_dynamicsWorld->removeRigidBody(base_ball);
            base_ball->setDamping(0,0);
            m_dynamicsWorld->addRigidBody(base_ball,collisionFilterGroup,collisionFilterMask);
            //m_dynamicsWorld->addRigidBody(base_ball);

            base_ball_location = base_ball->getCenterOfMassPosition();
            base_ball_trans = base_ball->getCenterOfMassTransform();

            //base_ball->setLinearVelocity(btVector3(-obj_speed_list[0],-obj_speed_list[1],-obj_speed_list[2]));
            //base_ball->setAngularVelocity(btVector3(0,0,0));
            m_allbaseballs.push_back(base_ball);

            // Special spring for base ball and base box unit 
            btTransform pivotInA(btQuaternion::getIdentity(),btVector3(0, radius, 0));						//par body's COM to cur body's COM offset
            btTransform pivotInB(btQuaternion::getIdentity(),btVector3(0, -y_len_link, 0));							//cur body's COM to cur body's PIV offset
            btGeneric6DofSpring2Constraint* fixed = new btGeneric6DofSpring2Constraint(*base_ball, *base,pivotInA,pivotInB);
            fixed->setLinearLowerLimit(btVector3(0,0,0));
            fixed->setLinearUpperLimit(btVector3(0,0,0));
            fixed->setAngularLowerLimit(btVector3(0,-1,0));
            fixed->setAngularUpperLimit(btVector3(0,1,0));
            for (int indx_tmp=3;indx_tmp<6;indx_tmp++){
                fixed->enableSpring(indx_tmp, true);
                float tmp_stiff = base_ball_base_spring_stf + spring_stfperunit*(num_units - indx_unit -1);
                if (tmp_stiff<100) tmp_stiff = 100;
                fixed->setStiffness(indx_tmp, tmp_stiff);
            }
            fixed->setEquilibriumPoint();

            m_jointFeedback.push_back(addFeedbackForSpring(fixed));
            spring_strtindx.push_back(indx_unit);

            m_dynamicsWorld->addConstraint(fixed,true);
        }


        btVector3 curr_pre  = btVector3(0, pre_y, pre_z);
        btVector3 curr_next = btVector3(0, next_y, next_z);

        // Create more spings between distant units
        for (int spring_indx=0;spring_indx < every_spring.size();spring_indx++){
            int tmp_every_spring    = every_spring[spring_indx];
            int tmp_inter_spring    = inter_spring[spring_indx];

            if ((indx_unit>=tmp_every_spring) && (indx_unit % tmp_inter_spring==0)){

                int new_indx = indx_unit - tmp_every_spring;

                btVector3 inter_point = findLineInter(m_allpre[new_indx], m_allnext[new_indx], curr_pre, curr_next);

                btTransform pivotInA(btQuaternion::getIdentity(),btVector3(0,  inter_point.distance((m_allpre[new_indx] + m_allnext[new_indx])/2), 0));
                btTransform pivotInB(btQuaternion::getIdentity(),btVector3(0, -inter_point.distance((curr_pre + curr_next)/2), 0));

                btGeneric6DofSpring2Constraint* fixed;

                fixed = new btGeneric6DofSpring2Constraint(*m_allbones[new_indx], *base, pivotInA,pivotInB);

                for (int indx_tmp=0;indx_tmp<6;indx_tmp++){
                    fixed->enableSpring(indx_tmp, true);
                    float tmp_stiff = base_spring_stiffness[spring_indx] + (num_units - indx_unit)*spring_stfperunit_list[spring_indx];
                    if (tmp_stiff<100) tmp_stiff = 100;
                    fixed->setStiffness(indx_tmp, tmp_stiff);
                }
                fixed->setEquilibriumPoint(3, -(deg_away - m_alldeg[new_indx]));
                

                m_jointFeedback.push_back(addFeedbackForSpring(fixed));
                spring_strtindx.push_back(new_indx);
                m_dynamicsWorld->addConstraint(fixed,true);
            }
        }

        m_allbones.push_back(base);
        m_allcentpos.push_back(base->getCenterOfMassPosition());
        m_allpre.push_back(curr_pre);
        m_allnext.push_back(curr_next);
        m_alldeg.push_back(deg_away);

        pre_unit    = base;
        pre_y       = next_y;
        pre_z       = next_z;
        pre_deg     = deg_away;
    }

    //cout << pre_z << endl;
    m_allbones_big_list.push_back(m_allbones);
    m_jointFeedback_big_list.push_back(m_jointFeedback);
    m_allcentpos_big_list.push_back(m_allcentpos);
    m_spring_strtindx.push_back(spring_strtindx);
}

void TestHingeTorque::loadParametersEveryWhisker(string curr_file_name, po::options_description desc_each){
    vector<float> base_spring_stiffness = {520};
    vector<float> spring_stfperunit_list = {1000};
    float basic_str     = 3000;
    float base_ball_base_spring_stf = 3000;
    float spring_stfperunit     = 1000;
    float linear_damp   = 0.77;
    float ang_damp      = 0.79;

    po::variables_map vm;
    
    try {
        ifstream ifs(curr_file_name);
        po::store(po::parse_config_file(ifs, desc_each), vm);
        po::notify(vm);

        if (vm.count("linear_damp")){
            linear_damp     = vm["linear_damp"].as<float>();
        }
        if (vm.count("ang_damp")){
            ang_damp        = vm["ang_damp"].as<float>();
        }

        if (vm.count("base_spring_stiffness")){
            base_spring_stiffness   = vm["base_spring_stiffness"].as< vector<float> >();
        }
        if (vm.count("spring_stfperunit_list")){
            spring_stfperunit_list   = vm["spring_stfperunit_list"].as< vector<float> >();
        }

        if ((inter_spring.size()!=base_spring_stiffness.size()) || (base_spring_stiffness.size()!=spring_stfperunit_list.size())){
            cerr << "error: spring related size not equal!" << endl;
            cerr << inter_spring.size() << " " << base_spring_stiffness.size() << " " << spring_stfperunit_list.size() << endl;
            exit(0);
        }

        if (vm.count("basic_str")){
            basic_str       = vm["basic_str"].as<float>();
        }
        if (vm.count("base_ball_base_spring_stf")){
            base_ball_base_spring_stf   = vm["base_ball_base_spring_stf"].as<float>();
        }
        if (vm.count("spring_stfperunit")){
            spring_stfperunit   = vm["spring_stfperunit"].as<float>();
        }

    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
    }

    m_base_spring_stiffness.push_back(base_spring_stiffness);
    m_spring_stfperunit_list.push_back(spring_stfperunit_list);
    m_basic_str.push_back(basic_str);
    m_base_ball_base_spring_stf.push_back(base_ball_base_spring_stf);
    m_spring_stfperunit.push_back(spring_stfperunit);
    m_linear_damp.push_back(linear_damp);
    m_ang_damp.push_back(ang_damp);

}

btRigidBody* TestHingeTorque::addObjasRigidBody(string fileName,
    float scaling[4], float orn[4], float pos[4], float mass_want, float control_len_now , int indx_obj){

    GLInstanceGraphicsShape* glmesh = LoadMeshFromObj(fileName.c_str(), "");
    printf("[INFO] Obj loaded: Extracted %d verticed from obj file [%s]\n", glmesh->m_numvertices, fileName.c_str());

    const GLInstanceVertex& v = glmesh->m_vertices->at(0);
    btConvexHullShape* shape = new btConvexHullShape((const btScalar*)(&(v.xyzw[0])), glmesh->m_numvertices, sizeof(GLInstanceVertex));

    //shape->optimizeConvexHull();

    int num_point = shape->getNumPoints();

    if (control_len_now!=-1){
        btVector3* all_point_list = shape->getUnscaledPoints();
        btVector3 min_vec, max_vec;

        for (int i=0;i<num_point;i++){
            btVector3 curr_point = all_point_list[i];

            for (int j=0;j<3;j++){
                if ((i==0) || (curr_point[j] < min_vec[j]))
                    min_vec[j] = curr_point[j];
                if ((i==0) || (curr_point[j] > max_vec[j]))
                    max_vec[j] = curr_point[j];
            }
        }

        float desire_scale = control_len_now/(max_vec - min_vec).norm();
        scaling[0] = desire_scale;
        scaling[1] = desire_scale;
        scaling[2] = desire_scale;
        obj_scaling_list[indx_obj*4] = desire_scale;
        obj_scaling_list[indx_obj*4 + 1] = desire_scale;
        obj_scaling_list[indx_obj*4 + 2] = desire_scale;
        obj_scaling_list[indx_obj*4 + 3] = control_len_now;

        mass_want = mass_want*pow((desire_scale/10), 3);
        obj_mass_list[indx_obj] = mass_want;
    }

    btVector3 localScaling(scaling[0],scaling[1],scaling[2]);
    shape->setLocalScaling(localScaling);

    btTransform startTransform;
    startTransform.setIdentity();

    float color[4] = {1,1,1,1};
    startTransform.setOrigin(btVector3(0,0,0));
    startTransform.setRotation(btQuaternion(orn[0],orn[1],orn[2],orn[3]));

    btVector3 position(pos[0],pos[1],pos[2]);

    if (reset_speed==1){
        btVector3 new_speed=cent_of_whisker_array - btVector3(pos[0], pos[1], pos[2]);
        new_speed.normalize();
        new_speed = new_speed*(btVector3(obj_speed_list[0], obj_speed_list[1], obj_speed_list[2]).norm());
        obj_speed_list[indx_obj*3] = new_speed[0];
        obj_speed_list[indx_obj*3 + 1] = new_speed[1];
        obj_speed_list[indx_obj*3 + 2] = new_speed[2];
    }

    if (reset_pos==1){

        btVector3 diff_vec = position - cent_of_whisker_array;
        float min_value = 0;
        int min_indx    = 0;

        for (int i=0;i<num_point;i++){
            btVector3 curr_point = shape->getScaledPoint(i);
            curr_point = startTransform(curr_point);
            float curr_value = curr_point.dot(diff_vec);
            if ((i==0) || (curr_value < min_value)){
                min_value   = curr_value;
                min_indx    = i;
            }
        }

        position        = position - startTransform(shape->getScaledPoint(min_indx));
    }
    startTransform.setOrigin(position);

    if (avoid_coll==1){
        float max_value = 0;
        btVector3 min_vec, max_vec;

        for (int i=0;i<num_point;i++){
            btVector3 curr_point = shape->getScaledPoint(i);

            for (int j=0;j<3;j++){
                if ((i==0) || (curr_point[j] < min_vec[j]))
                    min_vec[j] = curr_point[j];
                if ((i==0) || (curr_point[j] > max_vec[j]))
                    max_vec[j] = curr_point[j];
            }
        }

        btVector3 diff_vec = max_vec - min_vec;

        for (int i=0;i<20;i++)
            for (int j=0;j<20;j++)
                for (int k=0;k<20;k++){
                    btVector3 curr_point = btVector3(diff_vec[0]/20*i + min_vec[0], diff_vec[1]/20*j + min_vec[1], diff_vec[2]/20*k + min_vec[2]);
                    curr_point = startTransform(curr_point);

                    for (int jj=0;jj<m_allcentpos_big_list.size();jj++){
                        btVector3 now_dest = m_allcentpos_big_list[jj][0];

                        float time_need = (now_dest[1] - curr_point[1])/obj_speed_list[1];
                        btVector3 next_point = curr_point + time_need*btVector3(obj_speed_list[0], obj_speed_list[1], obj_speed_list[2]);
                        if ((next_point[2] > now_dest[2] - avoid_coll_z_off) && ( abs(next_point[0] - now_dest[0]) < avoid_coll_x_off) ){
                            if (max_value < next_point[2] - (now_dest[2] - avoid_coll_z_off))
                                max_value = next_point[2] - (now_dest[2] - avoid_coll_z_off);
                        }
                    }
                }

        /*
        for (int i=0;i<num_point;i++){
            btVector3 curr_point = shape->getScaledPoint(i);
            curr_point = startTransform(curr_point);
            
            for (int j=0;j<m_allcentpos_big_list.size();j++){
                btVector3 now_dest = m_allcentpos_big_list[j][0];

                float time_need = (now_dest[1] - curr_point[1])/obj_speed_list[1];
                btVector3 next_point = curr_point + time_need*btVector3(obj_speed_list[0], obj_speed_list[1], obj_speed_list[2]);
                if ((next_point[2] > now_dest[2] - avoid_coll_z_off) && ( abs(next_point[0] - now_dest[0]) < avoid_coll_x_off) ){
                    if (max_value < next_point[2] - (now_dest[2] - avoid_coll_z_off))
                        max_value = next_point[2] - (now_dest[2] - avoid_coll_z_off);
                }
            }
        }
        */

        startTransform.setOrigin(btVector3(position[0], position[1], position[2] - max_value));
    }

    // Get bounding box, center pos, store them
    {

        m_objstartTrans.push_back(startTransform);

        btVector3 min_vec, max_vec;

        for (int i=0;i<num_point;i++){
            btVector3 curr_point = shape->getScaledPoint(i);

            for (int j=0;j<3;j++){
                if ((i==0) || (curr_point[j] < min_vec[j]))
                    min_vec[j] = curr_point[j];
                if ((i==0) || (curr_point[j] > max_vec[j]))
                    max_vec[j] = curr_point[j];
            }
        }

        m_objboundingbox.push_back(max_vec - min_vec);
        m_objcenter.push_back((max_vec + min_vec)/2);

    }

    //btTriangleMesh* meshInterface = new btTriangleMesh();
    //for (int i=0;i<num_point/3;i++)
    //    meshInterface->addTriangle(shape->getScaledPoint(i*3), shape->getScaledPoint(i*3+1), shape->getScaledPoint(i*3+2));
    //btBvhTriangleMeshShape* trimesh = new btBvhTriangleMeshShape(meshInterface,true,true);

    //btRigidBody* body = createRigidBody(mass,startTransform,shape);
    //btRigidBody* body = createRigidBody(0.f,startTransform,trimesh);

    btCollisionShape* shape_compound = LoadShapesFromObj(fileName.c_str(), "", btVector3(scaling[0], scaling[1], scaling[2]));
    btRigidBody* body = createRigidBody(mass_want, startTransform, shape_compound);

    obj_pos_list[indx_obj*4] = startTransform.getOrigin()[0];
    obj_pos_list[indx_obj*4+1] = startTransform.getOrigin()[1];
    obj_pos_list[indx_obj*4+2] = startTransform.getOrigin()[2];

    int shapeId = m_guiHelper->registerGraphicsShape(&glmesh->m_vertices->at(0).xyzw[0], 
                                                                    glmesh->m_numvertices, 
                                                                    &glmesh->m_indices->at(0), 
                                                                    glmesh->m_numIndices,
                                                                    B3_GL_TRIANGLES, -1);
    //shape->setUserIndex(shapeId);
    int renderInstance = m_guiHelper->registerGraphicsInstance(shapeId,pos,orn,color,scaling);
    body->setUserIndex(renderInstance);

    return body;
}

void TestHingeTorque::initPhysics(){
	int upAxis = 1;
    pass_time   = 0;
    loss_ret    = 0;
    min_dis     = 10000;
    m_allbones_big_list.clear();
    m_allcentpos_big_list.clear();
    m_allbaseballs.clear();
    m_allobjs.clear();
    m_spring_strtindx.clear();
    m_objcenter.clear();
    m_objstartTrans.clear();
    m_objboundingbox.clear();

    all_force_data.clear();
    all_torque_data.clear();

	//BulletURDFImporter u2b(m_guiHelper, 0);
	
	//bool loadOk = u2b.loadURDF("/Users/chengxuz/barrel/bullet/bullet3/data/teddy_vhacd_2.urdf");

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("x_pos_base", po::value<vector<float>>()->multitoken(), "Position x of base")
        ("y_pos_base", po::value<vector<float>>()->multitoken(), "Position y of base")
        ("z_pos_base", po::value<vector<float>>()->multitoken(), "Position z of base")
        ("const_numLinks", po::value<vector<int>>()->multitoken(), "Number of units")
        ("qua_a_list", po::value<vector<float>>()->multitoken(), "Quadratic Coefficient")
        ("yaw_y_base", po::value<vector<float>>()->multitoken(), "Parameter yaw for btQuaternion")
        ("pitch_x_base", po::value<vector<float>>()->multitoken(), "Parameter pitch for btQuaternion")
        ("roll_z_base", po::value<vector<float>>()->multitoken(), "Parameter roll for btQuaternion")
        ("x_len_link", po::value<float>(), "Size x of cubes")
        ("y_len_link", po::value<float>(), "Size y of cubes")
        ("z_len_link", po::value<float>(), "Size z of cubes")
        ("radius", po::value<float>(), "Size of the ball at top")
        ("time_leap", po::value<float>(), "Time unit for simulation")
        ("inter_spring", po::value<vector<int>>()->multitoken(), "Number of units between two strings")
        ("every_spring", po::value<vector<int>>()->multitoken(), "Number of units between one strings")

        ("whisker_config_name", po::value<vector<string>>()->multitoken(), "Name of config files for each whisker")

        ("camera_dist", po::value<float>(), "Distance of camera")
        ("camera_yaw", po::value<float>(), "Distance of camera")
        ("camera_pitch", po::value<float>(), "Distance of camera")

        ("add_objs", po::value<int>(), "Whether to add objects, default is 0, not adding, 1 for adding")
        ("obj_filename", po::value<vector<string>>()->multitoken(), "Name of .obj files for each objects")
        ("obj_scaling_list", po::value<vector<float>>()->multitoken(), "Object scaling list")
        ("obj_pos_list", po::value<vector<float>>()->multitoken(), "Object position list")
        ("obj_orn_list", po::value<vector<float>>()->multitoken(), "Object orientation list")
        ("obj_speed_list", po::value<vector<float>>()->multitoken(), "Object speed list")
        ("offset_center_pos", po::value<vector<float>>()->multitoken(), "offset to the center point, only used when reset_pos is 1")
        ("obj_mass_list", po::value<vector<float>>()->multitoken(), "Object mass list")
        ("control_len", po::value<vector<float>>()->multitoken(), "Object list of whether to control the maximal length")
        ("reset_pos", po::value<int>(), "Whether to reset positions according to the center of whisker array, default is 1, resetting, 0 for not resetting")
        ("reset_speed", po::value<int>(), "Whether to reset speed according to the center of whisker array (the norm of input speed will be kept), default is 0, not resetting, 1 for resetting")
        ("avoid_coll", po::value<int>(), "Whether to reset position again to avoid collision with base balls, default is 0, not resetting, 1 for resetting")
        ("avoid_coll_z_off", po::value<float>(), "Z offset used during collision calculationg, default is 0")
        ("avoid_coll_x_off", po::value<float>(), "X offset used during collision calculationg for whether it's too far away, default is 5")
        ("get_normal", po::value<int>(), "Whether to get the normal picture from the center of whisker array to the center of obj (while they are at the same y position), default is 0 for no, 1 for getting")

        ("do_save", po::value<int>(), "Whether to save to hdf5, default is 0, not saving, 1 for saving to hdf5")
        ("FILE_NAME", po::value<string>(), "The filename for hdf5")
        ("num_unit_to_save", po::value<int>(), "The responses of string with starting index less than this number will be saved")
        ("sample_rate", po::value<int>(), "Sampling rate for saving, relative to time_leap")
        ("group_name", po::value<string>(), "Name for the group of dataset")

        ("time_limit", po::value<float>(), "Time limit for recording")
        ("initial_str", po::value<float>(), "Initial strength of force applied")
        ("max_str", po::value<float>(), "Max strength of force applied, used when force_mode=1")
        ("initial_stime", po::value<float>(), "Initial time to apply force")
        ("initial_poi", po::value<int>(), "Unit to apply the force")
        ("flag_time", po::value<int>(), "Whether open time limit")
        ("limit_softness", po::value<float>(), "Softness of the hinge limit")
        ("limit_bias", po::value<float>(), "Bias of the hinge limit")
        ("limit_relax", po::value<float>(), "Relax of the hinge limit")
        ("limit_low", po::value<float>(), "Lower bound of the hinge limit")
        ("limit_up", po::value<float>(), "Up bound of the hinge limit")
        ("velo_ban_limit", po::value<float>(), "While flag_time is 2, used for linear velocities of rigid bodys to judge whether stop")
        ("angl_ban_limit", po::value<float>(), "While flag_time is 2, used for angular velocities of rigid bodys to judge whether stop")
        ("force_limit", po::value<float>(), "While flag_time is 2, used for forces of rigid bodys to judge whether stop")
        ("torque_limit", po::value<float>(), "While flag_time is 2, used for torques of rigid bodys to judge whether stop")
        ("dispos_limit", po::value<float>(), "While flag_time is 2, used for distance to balance states of rigid bodys to judge whether stop")
        ("test_mode", po::value<int>(), "Whether enter test mode for some temp test codes, default is 0")
        ("force_mode", po::value<int>(), "Force mode to apply at the beginning, default is 0")
    ;

    po::options_description desc_each("Allowed options for each whisker");
    desc_each.add_options()
        ("linear_damp", po::value<float>(), "Control the linear damp ratio")
        ("ang_damp", po::value<float>(), "Control the angle damp ratio")
        ("basic_str", po::value<float>(), "Minimal strength of hinge's recover force")
        ("base_ball_base_spring_stf", po::value<float>(), "Base stiffness of spring for spring between base ball and first unit")
        ("spring_stfperunit", po::value<float>(), "Coefficient of stiffness of spring for index of units, only applied to springs between adjacent units")
        ("base_spring_stiffness", po::value<vector<float>>(), "Base stiffness of spring, only applied to springs between distant units")
        ("spring_stfperunit_list", po::value<vector<float>>(), "Coefficient of stiffness of spring for index of units, only applied to springs between distant units")
    ;

    po::variables_map vm;
    
    //b3Printf("Config name = %s",m_guiHelper->getconfigname());
    try {
        const char* file_name = m_guiHelper->getconfigname();
        ifstream ifs(file_name);
        po::store(po::parse_config_file(ifs, desc), vm);
        po::notify(vm);    

        if (vm.count("x_pos_base")){
            x_pos_base      = vm["x_pos_base"].as< vector<float> >();
        }
        if (vm.count("y_pos_base")){
            y_pos_base      = vm["y_pos_base"].as< vector<float> >();
        }
        if (vm.count("z_pos_base")){
            z_pos_base      = vm["z_pos_base"].as< vector<float> >();
        }
        if (vm.count("const_numLinks")){
            const_numLinks  = vm["const_numLinks"].as< vector<int> >();
        }
        if (vm.count("qua_a_list")){
            qua_a_list      = vm["qua_a_list"].as< vector<float> >();
        }
        if (vm.count("yaw_y_base")){
            yaw_y_base      = vm["yaw_y_base"].as< vector<float> >();
        }
        if (vm.count("pitch_x_base")){
            pitch_x_base    = vm["pitch_x_base"].as< vector<float> >();
        }
        if (vm.count("roll_z_base")){
            roll_z_base     = vm["roll_z_base"].as< vector<float> >();
        }
        if (vm.count("whisker_config_name")){
            whisker_config_name      = vm["whisker_config_name"].as< vector<string> >();
        }

        if ((x_pos_base.size()!=y_pos_base.size()) || (y_pos_base.size()!=z_pos_base.size()) || (x_pos_base.size()!=const_numLinks.size()) || (x_pos_base.size()!=qua_a_list.size()) || (x_pos_base.size()!=whisker_config_name.size())){
            cerr << "error: size not equal for (xyz,num)!" << endl;
            exit(0);
        }

        if (vm.count("x_len_link")){
            x_len_link      = vm["x_len_link"].as<float>();
        }
        if (vm.count("y_len_link")){
            y_len_link      = vm["y_len_link"].as<float>();
        }
        if (vm.count("z_len_link")){
            z_len_link      = vm["z_len_link"].as<float>();
        }
        if (vm.count("radius")){
            radius          = vm["radius"].as<float>();
        }

        if (vm.count("time_leap")){
            time_leap       = vm["time_leap"].as<float>();
        }

        if (vm.count("inter_spring")){
            inter_spring    = vm["inter_spring"].as< vector<int> >();
        }
        if (vm.count("every_spring")){
            every_spring    = vm["every_spring"].as< vector<int> >();
        }

        if ((every_spring.size()!=inter_spring.size())){
            cerr << "error: spring related size not equal!" << endl;
            exit(0);
        }

        if (vm.count("camera_dist")){
            camera_dist     = vm["camera_dist"].as<float>();
        }
        if (vm.count("camera_yaw")){
            camera_yaw     = vm["camera_yaw"].as<float>();
        }
        if (vm.count("camera_pitch")){
            camera_pitch     = vm["camera_pitch"].as<float>();
        }

        if (vm.count("add_objs")){
            add_objs        = vm["add_objs"].as<int>();
        }
        if (vm.count("obj_filename")){
            obj_filename    = vm["obj_filename"].as< vector<string> >();
        }
        if (vm.count("obj_scaling_list")){
            obj_scaling_list    = vm["obj_scaling_list"].as< vector<float> >();
        }
        if (vm.count("obj_pos_list")){
            obj_pos_list    = vm["obj_pos_list"].as< vector<float> >();
        }
        if (vm.count("obj_orn_list")){
            obj_orn_list    = vm["obj_orn_list"].as< vector<float> >();
        }
        if (vm.count("obj_speed_list")){
            obj_speed_list    = vm["obj_speed_list"].as< vector<float> >();
        }
        if (vm.count("offset_center_pos")){
            offset_center_pos    = vm["offset_center_pos"].as< vector<float> >();
        }
        if (vm.count("obj_mass_list")){
            obj_mass_list     = vm["obj_mass_list"].as< vector<float> >();
        }
        if (vm.count("control_len")){
            control_len         = vm["control_len"].as< vector<float> >();
        }
        if (vm.count("reset_pos")){
            reset_pos           = vm["reset_pos"].as<int>();
        }
        if (vm.count("reset_speed")){
            reset_speed         = vm["reset_speed"].as<int>();
        }
        if (vm.count("avoid_coll")){
            avoid_coll          = vm["avoid_coll"].as<int>();
        }
        if (vm.count("get_normal")){
            get_normal          = vm["get_normal"].as<int>();
        }
        if (vm.count("avoid_coll_z_off")){
            avoid_coll_z_off    = vm["avoid_coll_z_off"].as<float>();
        }
        if (vm.count("avoid_coll_x_off")){
            avoid_coll_x_off    = vm["avoid_coll_x_off"].as<float>();
        }


        if ((add_objs==1) and ((obj_scaling_list.size()!=4*obj_filename.size()) || (obj_speed_list.size()!=3*obj_filename.size())
                   || (obj_pos_list.size()!=obj_orn_list.size()) || (obj_orn_list.size()!=obj_scaling_list.size()) || 
                   (obj_filename.size()!=obj_mass_list.size()) || (obj_filename.size()!=control_len.size()))){
            cerr << "error: obj related size not equal!" << endl;
            exit(0);
        }

        if (vm.count("do_save")){
            do_save         = vm["do_save"].as<int>();
        }
        if (vm.count("FILE_NAME")){
            FILE_NAME       = vm["FILE_NAME"].as<string>();
        }
        if (vm.count("num_unit_to_save")){
            num_unit_to_save    = vm["num_unit_to_save"].as<int>();
        }
        if (vm.count("sample_rate")){
            sample_rate         = vm["sample_rate"].as<int>();
        }
        if (vm.count("group_name")){
            group_name          = vm["group_name"].as<string>();
        }

        if (vm.count("time_limit")){
            time_limit      = vm["time_limit"].as<float>();
        }
        if (vm.count("initial_str")){
            initial_str     = vm["initial_str"].as<float>();
        }
        if (vm.count("max_str")){
            max_str         = vm["max_str"].as<float>();
        }
        if (vm.count("initial_stime")){
            initial_stime   = vm["initial_stime"].as<float>();
        }
        if (vm.count("initial_poi")){
            initial_poi     = vm["initial_poi"].as<int>();
        }
        if (vm.count("flag_time")){
            flag_time       = vm["flag_time"].as<int>();
        }

        if (vm.count("limit_softness")){
            limit_softness  = vm["limit_softness"].as<float>();
        }
        if (vm.count("limit_bias")){
            limit_bias      = vm["limit_bias"].as<float>();
        }
        if (vm.count("limit_relax")){
            limit_relax     = vm["limit_relax"].as<float>();
        }
        if (vm.count("limit_low")){
            limit_low       = vm["limit_low"].as<float>();
        }
        if (vm.count("limit_up")){
            limit_up        = vm["limit_up"].as<float>();
        }

        if (vm.count("velo_ban_limit")){
            velo_ban_limit  = vm["velo_ban_limit"].as<float>();
        }
        if (vm.count("angl_ban_limit")){
            angl_ban_limit  = vm["angl_ban_limit"].as<float>();
        }
        if (vm.count("force_limit")){
            force_limit     = vm["force_limit"].as<float>();
        }
        if (vm.count("torque_limit")){
            torque_limit    = vm["torque_limit"].as<float>();
        }
        if (vm.count("dispos_limit")){
            dispos_limit    = vm["dispos_limit"].as<float>();
        }

        if (vm.count("test_mode")){
            test_mode       = vm["test_mode"].as<int>();
        }

        if (vm.count("force_mode")){
            force_mode      = vm["force_mode"].as<int>();
        }

    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "123123123\n";
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
    }

    for (int indx_unit=0;indx_unit<whisker_config_name.size();indx_unit++){
        loadParametersEveryWhisker(whisker_config_name[indx_unit], desc_each);
    }
    
	m_guiHelper->setUpAxis(upAxis);

	createEmptyDynamicsWorld();
	m_dynamicsWorld->getSolverInfo().m_splitImpulse = false;
	
    m_dynamicsWorld->setGravity(btVector3(0,0,0));
    
	m_guiHelper->createPhysicsDebugDrawer(m_dynamicsWorld);
	int mode = 	btIDebugDraw::DBG_DrawWireframe
				+btIDebugDraw::DBG_DrawConstraints
				+btIDebugDraw::DBG_DrawConstraintLimits;

    if (m_guiHelper->have_visualize!=100) {
        m_dynamicsWorld->getDebugDrawer()->setDebugMode(mode);
    }


    btVector3 linkHalfExtents(x_len_link, y_len_link, z_len_link);

    btBoxShape* baseBox = new btBoxShape(linkHalfExtents);
    btBoxShape* linkBox1 = new btBoxShape(linkHalfExtents);
    btSphereShape* linkSphere = new btSphereShape(radius);

    if (test_mode==1) {

        for (int big_list_indx=0;big_list_indx < const_numLinks.size(); big_list_indx++) { // create one single whisker 
            //addQuaUnits(qua_a_list[big_list_indx], 0, 0, const_numLinks[big_list_indx], btTransform(btQuaternion( yaw_y_base[big_list_indx], pitch_x_base[big_list_indx], roll_z_base[big_list_indx]),btVector3(x_pos_base[big_list_indx], y_pos_base[big_list_indx], z_pos_base[big_list_indx])));
            float yaw_now = yaw_y_base[big_list_indx];
            float pitch_now = pitch_x_base[big_list_indx];
            float roll_now = roll_z_base[big_list_indx];
            //btTransform tmp_trans(btQuaternion(0,0,-3.1415926/4), btVector3(0,0,0));
            btTransform tmp_trans(btQuaternion(0,0,0), btVector3(0,0,0));
            float c_x = cos(pitch_now), s_x = sin(pitch_now);
            float c_y = cos(yaw_now), s_y = sin(yaw_now);
            float c_z = cos(roll_now), s_z = sin(roll_now);
            float xx = c_y*c_z, xy = c_z*s_x*s_y - c_x*s_z, xz = s_x*s_z + c_x*c_z*s_y;
            float yx = c_y*s_z, yy = c_x*c_z + s_x*s_y*s_z, yz = c_x*s_y*s_z - c_z*s_x;
            float zx = -s_y, zy = c_y*s_x, zz = c_x*c_y;
            //btMatrix3x3 qua_mat(xx, xy, xz, yx, yy, yz, zx, zy, zz);
            btMatrix3x3 qua_mat(zz, zy, zx, yz, yy, yx, xz, xy, xx);
            //addQuaUnits(qua_a_list[big_list_indx], 0, 0, const_numLinks[big_list_indx], tmp_trans*btTransform(btQuaternion( yaw_y_base[big_list_indx], pitch_x_base[big_list_indx], roll_z_base[big_list_indx]),btVector3(x_pos_base[big_list_indx], y_pos_base[big_list_indx], z_pos_base[big_list_indx])));
            addQuaUnits(qua_a_list[big_list_indx], 0, 0, big_list_indx, tmp_trans*btTransform(qua_mat,btVector3(x_pos_base[big_list_indx], y_pos_base[big_list_indx], z_pos_base[big_list_indx])));
        }

    } else {
        for (int big_list_indx=0;big_list_indx < const_numLinks.size(); big_list_indx++) { // create one single whisker 
            //addQuaUnits(qua_a_list[big_list_indx], 0, 0, const_numLinks[big_list_indx], btTransform(btQuaternion( yaw_y_base[big_list_indx], pitch_x_base[big_list_indx], roll_z_base[big_list_indx]),btVector3(x_pos_base[big_list_indx], y_pos_base[big_list_indx], z_pos_base[big_list_indx])));
            float yaw_now = yaw_y_base[big_list_indx];
            float pitch_now = pitch_x_base[big_list_indx];
            float roll_now = roll_z_base[big_list_indx];
            //btTransform tmp_trans(btQuaternion(0,0,-3.1415926/4), btVector3(0,0,0));
            btTransform tmp_trans(btQuaternion(0,0,0), btVector3(0,0,0));
            float c_x = cos(pitch_now), s_x = sin(pitch_now);
            float c_y = cos(yaw_now), s_y = sin(yaw_now);
            float c_z = cos(roll_now), s_z = sin(roll_now);
            float xx = c_y*c_z, xy = c_z*s_x*s_y - c_x*s_z, xz = s_x*s_z + c_x*c_z*s_y;
            float yx = c_y*s_z, yy = c_x*c_z + s_x*s_y*s_z, yz = c_x*s_y*s_z - c_z*s_x;
            float zx = -s_y, zy = c_y*s_x, zz = c_x*c_y;
            //btMatrix3x3 qua_mat(xx, xy, xz, yx, yy, yz, zx, zy, zz);
            btMatrix3x3 qua_mat(zz, zy, zx, yz, yy, yx, xz, xy, xx);
            //addQuaUnits(qua_a_list[big_list_indx], 0, 0, const_numLinks[big_list_indx], tmp_trans*btTransform(btQuaternion( yaw_y_base[big_list_indx], pitch_x_base[big_list_indx], roll_z_base[big_list_indx]),btVector3(x_pos_base[big_list_indx], y_pos_base[big_list_indx], z_pos_base[big_list_indx])));
            addQuaUnits(qua_a_list[big_list_indx], 0, 0, big_list_indx, tmp_trans*btTransform(qua_mat,btVector3(x_pos_base[big_list_indx], y_pos_base[big_list_indx], z_pos_base[big_list_indx])));
        }

        if (add_objs==1){
            cal_cent_whisker();
            for (int indx_obj=0;indx_obj<obj_filename.size();indx_obj++){
                string fileName = obj_filename[indx_obj];

                float scaling[4] = {obj_scaling_list[indx_obj*4], obj_scaling_list[indx_obj*4+1], 
                    obj_scaling_list[indx_obj*4+2], obj_scaling_list[indx_obj*4+3]};

                float orn[4] = {obj_orn_list[indx_obj*4], obj_orn_list[indx_obj*4+1], 
                    obj_orn_list[indx_obj*4+2], obj_orn_list[indx_obj*4+3]};

                float pos[4] = {obj_pos_list[indx_obj*4], obj_pos_list[indx_obj*4+1], 
                    obj_pos_list[indx_obj*4+2], obj_pos_list[indx_obj*4+3]};

                float mass_want = obj_mass_list[indx_obj];

                float control_len_now = control_len[indx_obj];

                m_allobjs.push_back(addObjasRigidBody(fileName, scaling, orn, pos, mass_want, control_len_now, indx_obj));


                if (get_normal==1){
                    //cout << "Now getting normal" << endl;
                    btVector3 now_obj_center = m_objstartTrans[indx_obj](m_objcenter[indx_obj]);
                    float time_need = (orig_cent_of_whisker_array[1] - now_obj_center[1]) / obj_speed_list[1];
                    btVector3 new_from = orig_cent_of_whisker_array + time_need*btVector3(-obj_speed_list[0], -obj_speed_list[1], -obj_speed_list[2]);
                    btVector3 unit_x = now_obj_center - new_from;
                    float tmp_change = unit_x[0];
                    unit_x[0] = -unit_x[2];
                    unit_x[2] = tmp_change;

                    btVector3 new_from_2 = now_obj_center + unit_x;
                    btVector3 new_from_3 = now_obj_center - unit_x;
                    btVector3 new_from_4 = orig_cent_of_whisker_array + btVector3(0, unit_x.norm(), 0);
                    btVector3 new_from_5 = orig_cent_of_whisker_array - btVector3(0, unit_x.norm(), 0);
                        
                    unit_x.normalize();
                    unit_x = unit_x * normal_unit_len;
                    btVector3 unit_y = btVector3(0, normal_unit_len, 0);

                    btVector3 old_unit_x = unit_x;

                    all_normals.push_back(get_normal_picture(new_from, now_obj_center, normal_im_h, normal_im_w, unit_x, unit_y, control_len_now));

                    unit_x = now_obj_center - new_from;
                    unit_x.normalize();
                    unit_x = unit_x * normal_unit_len;
                    all_normals.push_back(get_normal_picture(new_from_2, now_obj_center, normal_im_h, normal_im_w, unit_x, unit_y, control_len_now));
                    all_normals.push_back(get_normal_picture(new_from_3, now_obj_center, normal_im_h, normal_im_w, -unit_x, unit_y, control_len_now));
                    all_normals.push_back(get_normal_picture(new_from_4, now_obj_center, normal_im_h, normal_im_w, unit_x, old_unit_x, control_len_now)); 
                    all_normals.push_back(get_normal_picture(new_from_5, now_obj_center, normal_im_h, normal_im_w, -unit_x, -old_unit_x, control_len_now)); 
                    //cout << "Getting normal finished!" << endl;

                }
            }
        }
        //set_initial_speed_all(btVector3(-obj_speed_list[0],-obj_speed_list[1],-obj_speed_list[2]));
    }
	
	m_guiHelper->autogenerateGraphicsObjects(m_dynamicsWorld);
}

class CommonExampleInterface*    TestHingeTorqueCreateFunc(CommonExampleOptions& options){
	return new TestHingeTorque(options.m_guiHelper);
}
