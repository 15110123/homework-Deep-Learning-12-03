# Tham khảo tại: https://blog.paperspace.com/implementing-gans-in-tensorflow/ 

import tensorflow as tf
import Generator as gen
import Discriminator as disc
import Data
import matplotlib.pyplot as plt
import numpy as np

X = tf.placeholder(tf.float32,[None,2])
Z = tf.placeholder(tf.float32,[None,2])

G_sample = gen.generator(Z)
r_logits, r_rep = disc.discriminator(X)
f_logits, g_rep = disc.discriminator(G_sample,reuse=True)

# Định nghĩa hàm loss. 
disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))

# Các tham chiếu sử dụng trong suốt quá trình session chạy được gọi là variable. 
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

# Định nghĩa optimizer là RMSProp, hiệu quả hơn so với back propagation truyền thống khi dùng với vanilla GAN. 
gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) # G Train step
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) # D Train step

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

batch_size = 256
nd_steps = 10
ng_steps = 10
x_plot = Data.sample_data(n=batch_size)

for i in range(10001):
    # Lấy dữ liệu X có n là kích thước batch. 
    X_batch = Data.sample_data(n=batch_size)

    # Tạo noise. 
    Z_batch = Data.sample_Z(batch_size, 2)

    # Huấn luyện discriminator. 
    for _ in range(nd_steps):
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})

    # Chú thích: 
    # rrep_dstep: Dữ liệu thật (output của h3) trước khi generate tại thời điểm i. 
    # grep_dstep: Dữ liệu giả (output của h3) trước khi generate tại thời điểm i. 
    rrep_dstep, grep_dstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

    # Huấn luyện generator. 
    for _ in range(ng_steps):
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

    # Chú thích: 
    # rrep_gstep: Dữ liệu thật (output của h3) sau khi generate tại thời điểm i. 
    # grep_gstep: Dữ liệu giả (output của h3) sau khi generate tại thời điểm i. 
    rrep_gstep, grep_gstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})
    if (i % 100 == 0):
        print ("Iterations: {} Discriminator loss: {:.2f} Generator loss: {:.2f}".format(i,dloss,gloss))
        
        plt.figure()

        # Chạy gen để ra output dữ liệu giả. 
        g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
        xax = plt.scatter(x_plot[:,0], x_plot[:,1])
        gax = plt.scatter(g_plot[:,0],g_plot[:,1])

        plt.legend((xax,gax), ("Dữ liệu thật","Dữ liệu giả"))
        plt.title('Samples at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig('plots/iterations/iteration_%d.png'%i)
        plt.close()

        plt.figure()
        rrd = plt.scatter(rrep_dstep[:,0], rrep_dstep[:,1], alpha=0.5)
        rrg = plt.scatter(rrep_gstep[:,0], rrep_gstep[:,1], alpha=0.5)
        grd = plt.scatter(grep_dstep[:,0], grep_dstep[:,1], alpha=0.5)
        grg = plt.scatter(grep_gstep[:,0], grep_gstep[:,1], alpha=0.5)


        plt.legend((rrd, rrg, grd, grg), ("Dữ liệu thật trước khi gen","Dữ liệu thật sau khi gen",
                               "Dữ liệu giả trước khi gen","Dữ liệu giả sau khi gen"))
        plt.title('Transformed Features at i = %d'%i)
        plt.tight_layout()
        plt.savefig('plots/features/feature_transform_%d.png'%i)
        plt.close()