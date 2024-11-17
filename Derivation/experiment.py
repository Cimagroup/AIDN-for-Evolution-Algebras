from auxiliary import *


#print("Write desire dimension:\n")
#d=int(input())
for d in range(2,10,2):
        print("----------------------------------")
        print("DIMENSION ",d)
        fs=tf.convert_to_tensor(np.identity(d), dtype='float32') #BASE FIJA
        k=d**2
        gs = get_n_operators(1,"sigmoid",False,k) # k of dimension 1

        # Relations
        def r1(gs,x):
            l= len(gs)
            n= int(np.sqrt(l))
            gs = np.array(gs).reshape((n,n))
            indexes = [(i,j) for i in range(n) for j in range(n)]
            r1_list = [[(gs[k,i](x)*gs[i,j](x)+gs[k,j](x)*gs[j,i](x)) for (i,j) in indexes if i!=j] 
                       for k in range(n)]
            return tf.reshape(tf.nest.flatten(r1_list),(l*(n-1),))

        def r2(gs,x):  
            l= len(gs)
            n= int(np.sqrt(l))
            gs = np.array(gs).reshape((n,n))
            indexes = [(i,j) for i in range(n) for j in range(n)]
            r2_list = [(2*gs[i,j](x)*gs[j,j](x)-tf.math.reduce_sum([gs[i,k](x)*gs[k,j](x) for k in range(n)]))
                       for (i,j) in indexes]
            return tf.reshape(tf.nest.flatten(r2_list),(l,))

        def r3(gs,x): # non-trivial solution
            l= len(gs)
            n= int(np.sqrt(l))
            gs = np.array(gs).reshape((n,n))
            r3_list = [tf.math.reduce_sum([1/(10*tf.math.reduce_sum([tf.abs(gs[i,j](x)) for i in range(n)])**2)]) for j in range(n)]
            return r3_list




        # Provide a model whose outputs are the relationships
        def group_rep_net(fs,gs,input_shape=1):
            input_tensor_1=Input(shape=(input_shape,))
            R_output = tf.concat([r1(gs,input_tensor_1),
                                  r2(gs,input_tensor_1),
                                  r3(gs,input_tensor_1)],axis=0)

            M= Model([input_tensor_1],R_output)
            return M

        M=group_rep_net(fs,gs,input_shape=1)
        loss=group_rep_loss(input_dim=1,d=d)

        n=1
        x_data=np.array([[1]])
        y_data = np.reshape(np.array([0.0]*10),(1,10))

        history =train_net(M,x_data,y_data,loss,1,0.01,50000)
        print("Number of needed epochs:",len(history.history['loss']))
        P=x_data[0]
        A=np.reshape(np.array([gs[i](P) for i in range(len(gs))]),(d,d))
        np.savetxt("structure_matrices/structure_matrix_"+str(d)+".txt",A)
        print("Structure matrix:\n",A)
        loss_rel=M.predict(x_data)
        print("Relation loss:",loss_rel)
        np.savetxt("loss/relation_values_"+str(d)+".txt",loss_rel)
