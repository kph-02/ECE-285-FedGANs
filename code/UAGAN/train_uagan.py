import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    
    # Set key parameters for HAM10000
    opt.n_class = 7  # HAM10000 has 7 classes
    opt.batch_size = 32
    opt.niter = 100  # Number of epochs
    opt.niter_decay = 100  # Number of decay epochs
    opt.lr_G = 0.0002  # Generator learning rate
    opt.lr_D = 0.0002  # Discriminator learning rate
    
    dataset = create_dataset(opt)  # create a dataset
    dataset_size = len(dataset)
    print('The number of training samples = %d' % dataset_size)
    
    model = create_model(opt)      # create a model
    model.setup(opt)               # regular setup: load and print networks
    visualizer = Visualizer(opt)   # create a visualizer
    total_iters = 0                # the total number of training iterations
    
    # Outer loop for different epochs
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        
        # Inner loop for training iterations
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            # Forward pass and optimization
            model.set_input(data)         # unpack data from dataset
            model.optimize_parameters()   # calculate loss and update weights
            
            # Display results periodically
            if total_iters % opt.display_freq == 0:   
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            
            # Print losses
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                
            # Save model periodically
            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            
            iter_data_time = time.time()
        
        # Save model at the end of epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        
        print('End of epoch %d / %d \t Time Taken: %d sec' % 
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch