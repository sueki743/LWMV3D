import math

from train import *


@jit
def box3d_transform_inv(et_boxes3d, deltas):
    num=len(et_boxes3d)
    boxes3d=np.zeros((num,8,3),dtype=np.float32)
    for n in range(num):
        e=et_boxes3d[n]
        center = np.sum(e,axis=0, keepdims=True)/8
        scale = (np.sum((e-center)**2)/8)**0.5

        d=deltas[n]
        boxes3d[n]= e+scale*d

    return boxes3d


def regularise_box3d(boxes3d):
    num = len(boxes3d)
    reg_boxes3d =np.zeros((num,8,3),dtype=np.float32)
    for n in range(num):
        b=boxes3d[n]

        dis=0
        corners = np.zeros((4,3),dtype=np.float32)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,k+4
            dis +=np.sum((b[i]-b[j])**2) **0.5
            corners[k] = (b[i]+b[j])/2

        dis = dis/4
        b = reg_boxes3d[n]
        for k in range(0,4):
            i,j=k,k+4
            b[i]=corners[k]-dis/2*np.array([0,0,1])
            b[j]=corners[k]+dis/2*np.array([0,0,1])

    return reg_boxes3d


def rcnn_nms(probs, deltas, rois3d, score_threshold=0.75, nms_threshold=0.001):
    cls=1  # do for class-one only
    probs = probs[:,cls] #see only class-1
    idx = np.where(probs>score_threshold)[0]

    #post processing
    rois3d = rois3d[idx]
    deltas = deltas[idx,cls]
    probs  = probs [idx]


    if deltas.shape[1:]==(8,3):
        boxes3d  = box3d_transform_inv(rois3d, deltas)
        boxes3d  = regularise_box3d(boxes3d)
        boxes=box3d_to_top_box(boxes3d)
        # dets = np.c_[boxes3d[:, 1, 0], boxes3d[:, 1, 1], boxes3d[:, 3, 0], boxes3d[:, 3, 1], probs]
        dets = np.c_[boxes, probs]
        # keep=np.logical_and((dets[:,0]<dets[:,2]),(dets[:,1]<dets[:,3]))
        # dets=dets[keep]
        keep = nms(dets, nms_threshold)
        return probs[keep], boxes3d[keep]


def boxes3d_decompose(boxes3d):
    # translation
    T_x = np.sum(boxes3d[:, 0:4, 0], 1) / 4.0
    T_y = np.sum(boxes3d[:, 0:4, 1], 1) / 4.0
    T_z = np.sum(boxes3d[:, 0:4, 2], 1) / 4.0

    Points0 = boxes3d[:, 0, 0:2]
    Points1 = boxes3d[:, 1, 0:2]
    Points2 = boxes3d[:, 2, 0:2]

    dis1=np.sum((Points0-Points1)**2,1)**0.5
    dis2=np.sum((Points1-Points2)**2,1)**0.5

    dis1_is_max=dis1>dis2

    #length width heigth
    L=np.maximum(dis1,dis2)
    W=np.minimum(dis1,dis2)
    H=np.sum((boxes3d[:,0,0:3]-boxes3d[:,4,0:3])**2,1)**0.5

    # rotation
    yaw=lambda p1,p2,dis: math.atan2(p2[1]-p1[1],p2[0]-p1[0])
    R_x = np.zeros(len(boxes3d))
    R_y = np.zeros(len(boxes3d))
    R_z = [yaw(Points0[i],Points1[i],dis1[i]) if is_max else  yaw(Points1[i],Points2[i],dis2[i])
           for is_max,i in zip(dis1_is_max,range(len(dis1_is_max)))]
    R_z=np.array(R_z)

    translation = np.c_[T_x,T_y,T_z]
    size = np.c_[H,W,L]
    rotation= np.c_[R_x,R_y,R_z]
    return translation,size,rotation


def infer(load_path, output_dir):
    loader = Loader(dirs=DIR_VAL, is_testset=True)
    net, top_view_anchors, anchors_inside_inds = make_net(*loader.get_shape())
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, load_path)
        os.makedirs(output_dir, exist_ok=True)
        print()
        time_start = time_prev = time()
        for i, (batch_rgb_images, batch_top_view, _, _, tag, calib, resize_coef) in enumerate(loader):
            fd1 = {
                IS_TRAIN_PHASE: False,
                net['top_view']: batch_top_view,
            }
            batch_proposals, batch_proposal_scores = sess.run((net['proposals'], net['proposal_scores']), fd1)
            # batch_proposal_scores = np.reshape(batch_proposal_scores, -1)  # unused
            top_rois = batch_proposals
            if len(top_rois) == 0:
                boxes3d, probs = np.zeros((0, 8, 3)), []
            else:
                rois3d = project_to_roi3d(top_rois)
                rgb_rois = project_to_rgb_roi(rois3d, calib, resize_coef)

                fd2 = {
                    IS_TRAIN_PHASE: False,
                    net['top_view']: batch_top_view,
                    net['rgb_images']: batch_rgb_images,
                    net['top_rois']: top_rois,
                    net['rgb_rois']: rgb_rois,
                }
                fuse_probs, fuse_deltas = sess.run([net['fuse_probs'], net['fuse_deltas']], fd2)

                probs, boxes3d = rcnn_nms(fuse_probs, fuse_deltas, rois3d, score_threshold=0.5)

            # output
            with open(os.path.join(output_dir, tag+'.txt'), 'w') as file:
                if len(boxes3d) != 0:
                    translation, size, rotation = boxes3d_decompose(boxes3d[:, :, :])
                    boxes3d_im = calib.velo_to_im(boxes3d)
                    translation_cam = calib.velo_to_cam(translation)
                    for j in range(len(boxes3d)):
                        label = Label()
                        label.type = 'Car'
                        label.bbox = np.array([
                            boxes3d_im[j, :, 0].min(),
                            boxes3d_im[j, :, 1].min(),
                            boxes3d_im[j, :, 0].max(),
                            boxes3d_im[j, :, 1].max(),
                        ])
                        label.dimensions = size[j]
                        label.location = translation_cam[j]
                        label.rotation_y = - np.pi/2 - rotation[j, 2]
                        label.score = probs[j]
                        label.write(file)

            if i % 100 == 99:
                time_now = time()
                print('100 frames took %.2f seconds.' % (time_now - time_prev))
                time_prev = time_now
        print('%d frames took %.2f seconds.' % (len(loader.tags), time() - time_start))


if __name__ == '__main__':
    time_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    infer(args.model_path, args.output_dir)
    print('infer.py took %.2f seconds.' % (time() - time_start))
