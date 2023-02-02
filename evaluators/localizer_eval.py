

def save_results_localizer(predictions, epoch, performence, save_path, mode):
        
    xs = [] 
    names = []
    ys_center_map = []
    yhats_center_map = [] 
    clips = []
    stacks = []

    
    line_width = 2 if self.config.scale_factor >1 else 1
    fontScale = 1 if self.config.scale_factor >1 else 0.3
    for i, data in enumerate(predictions): 
        
        x, y, y_hat = data   

        nonzero = torch.nonzero(x.sum(dim=[3,4]))[:,2]
        clips.append((nonzero.min(), nonzero.max()+1))  

        xs_ = []
        for x_ in x[0, 0]:
            x_ = np.uint8(255*np.repeat(x_[..., None].clone().numpy(), 3, axis=2))

            font = cv.FONT_HERSHEY_SIMPLEX  
            
            color = (255, 0, 0) 
            thickness = 2 if self.config.scale_factor >1 else 1
                
            x_ = cv.putText(x_, f'{y.name[0][7:]}', (5, 8*self.config.scale_factor) , font, fontScale, color, thickness, cv.LINE_AA)
            x_ = cv.putText(x_, f'{epoch}', (5, 16*self.config.scale_factor + 1*8*self.config.scale_factor) , font, fontScale, color, thickness, cv.LINE_AA) 
            
            value = performence['mse_error']
            str_ = f'mse_error: {value[i]}'
            x_ = cv.putText(x_, str_, (5, 16*self.config.scale_factor + 2*8*self.config.scale_factor) , font, fontScale, color, thickness, cv.LINE_AA) 
            xs_ += [x_[None]]
        xs_ = np.concatenate(xs_, axis=0)
            
        xs.append(xs_)  

        if y_hat.voxel is not None: 
            ys_center_map.append(y.voxel)
            yhats_center_map.append(y_hat.voxel)

    
    xs = np.concatenate(xs, axis=0) 
    ys_center_map = torch.cat(ys_center_map, dim=1).cpu().numpy()[0]
    yhats_center_map = torch.cat(yhats_center_map, dim=1).cpu().numpy()[0]
        
    # y = np.uint8(ys_center_map) + 2 * np.uint8(yhats_center_map)
    
    ys_overlap = blend_cpu2(torch.from_numpy(xs).float(), torch.from_numpy(ys_center_map), 4, factor=0.5) # *2 is hack to get the overlay the color red
    yhats_overlap = blend_cpu2(torch.from_numpy(xs).float(), torch.from_numpy(yhats_center_map), 4, factor=0.5) # *2 is hack to get the overlay the color red
        
    overlay = np.concatenate([ys_overlap, yhats_overlap], axis=2)
    # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/pred_voxels.tif', np.uint8(y_overlay))
    io.imsave(save_path + mode + 'overlay_y_hat.tif', overlay) 