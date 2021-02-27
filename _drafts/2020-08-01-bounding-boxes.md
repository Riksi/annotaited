---
layout: post
title:  "A guide to bounding boxes for object detection"
date:   2020-08-01 16:46:47
categories: jekyll update
---

## Bounding boxes 

In the simplest form you locate an object within an image (or a volume) by drawing a box around it. Bounding boxes can be represented in a number ways. We will focus on the $xyxy$ format where the box is represented by its top-left and bottom-right corner co-ordinates $[x_1, y_1, x_2, y_2]$ but will also sometimes use the $whxy$ format where we specify the width, height, and $x$ and $y$ coordinates of the centre, $[w, h, x_\text{centre}, y_\text{centre}]$. 

Note that in our definition $x_2$ and $y_2$ will be *inside* the box. But these can vary in elsewhere. Usually it won't make a huge difference but will cause problems with boxes at the edge of an image. 

Can you illustrate the relationship between the bounding box, its centre and dimensions? 

<div class="collapse-bbox_img">
<div markdown="1">
![png]({{ site.baseurl }}/assets/ODD_Boxes/bbox.jpg)
</div>
<p class='textblock'>
Note that the coordinates $[x_1, y_1, x_2, y_2]$ are *inside* the bounding box and are located at the centre of a unit square.
</p>
</div>

Let us now code this. From the start we will work with arbitrary sized tensors of boxes. These can have any number of dimensions and the only rule is that the size of the last dimension is 4. 

<div class='collapse-to_whxy'>
<div markdown="1">
```python
def to_whxy(box):
    """Return width, height, x centre, and y centre for an
    array of boxes.
    
    Adapted from https://github.com/facebookresearch/maskrcnn-benchmark
    
    box: An arbitrary dimensional tensor with a final dimension of size 4
    
    """
    
    # Add 1 to get dims since x2,y2 inside 
    w = box[..., 2] - box[..., 0] + 1
    h = box[..., 3] - box[..., 1] + 1
    x_ctr = box[..., 0] + 0.5 * (w - 1)
    y_ctr = box[..., 1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr 
```
</div>
</div>

To plot the bounding box on a grid via `plot_rectangle` we  draw a rectangle with top-left corner $(x_1 - 0.5, y_1 - 0.5)$ and bottom-right corner $(x_1 + w + 05, y_1 + w + 0.5) = (x_2 + 1.5, y_2 + 1.5)$, the $\pm 0.5$ shift because of the convention that $x, y$ and using $x2 + 1, y2 + 1$ as $x2, y2$ are assumed to be inside the box.

Let us write a function `plot_rectangle` that does given a bounding box `[x1, y1, x2, y2]` does just that.

<div class='collapse-plot_rectangle'>
<div markdown="1">
```python
def plot_rectangle(box, ax, clr='b', linewidth=1):
    box = np.array(box)
    x1, y1, x2, y2 = box
    w, h, _, _ = to_whxy(box)
    ax.add_artist(plt.Rectangle(xy=(x1 - 0.5, y1 - 0.5), height=h, width=w, fill=False, color=clr,
                               linewidth=linewidth))
    
```
</div>
</div>

Now we can plot some boxes

```python
fig, ax = plt.subplots(1, figsize=(6, 6))
plot_rectangle([0, 0, 3, 3], ax, linewidth=2)
ax.set_xlim([-1, 4.])
ax.set_ylim([-1, 4.])
ax.set_xticks(np.arange(-0.5, 4., 1));
ax.set_yticks(np.arange(-0.5, 4., 1));
ax.grid()
ax.plot([0, 3], [0, 3], 'x')
ax.invert_yaxis();
```

![png]({{ site.baseurl }}/assets/ODD_Boxes/output_7_0.png)

Note that `matplotlib` follows the convention above for plotting images. The documentation for `imshow` states:
> Pixels have unit size in data coordinates. Their centers are on
    integer coordinates, and their center coordinates range from 0 to
    columns-1 horizontally and from 0 to rows-1 vertically.
    
This means we can plot the bounding boxes on top in the same fashion.  

```python
img = plt.imread('hibiscus-3601937_1920.jpg')
img = resize(img, (600, 1000))
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img, origin='upper')
ax.axis('off');

bbox1 = [85,   0, 315, 220]
bbox2 = [330,   140, 650, 480]

plot_rectangle(np.stack(bbox1), ax=ax, clr='w', linewidth=3);
plot_rectangle(np.stack(bbox2), ax=ax, clr='w', linewidth=3);
```

![png]({{ site.baseurl }}/assets/ODD_Boxes/output_9_0.png)


```python
img2 = plt.imread('flowers-4365828_1920.jpg')
img2 = resize(img2, (600, 1000))
bbox3 = [130,   150, 505, 440]
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img2)
ax.axis('off');
plot_rectangle(np.stack(bbox3), ax=ax, clr='w', linewidth=3);
```


![png]({{ site.baseurl }}/assets/ODD_Boxes/output_10_0.png)


Note that $x$ corresponds to columns and that $y$ corresponds to rows. To crop a box from the image slice it between $x1$ and $x2 + 1 = x1 + w$ along the column dimension and $y1$ and $y2 + 1 = y1 + h$ along the row dimenson. This will give us the same box as box if we assume the pixel at index $(y=i, x=j)$ in the grid to be a unit square centred at $(x y)$.


```python
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 8))

crop1 = img[bbox1[1]:bbox1[3] + 1, bbox1[0]:bbox1[2] + 1]
crop2 = img[bbox2[1]:bbox2[3] + 1, bbox2[0]:bbox2[2] + 1]
crop3 = img2[bbox3[1]:bbox3[3] + 1, bbox3[0]:bbox3[2] + 1]

ax0.imshow(crop1)
ax1.imshow(crop2)
ax2.imshow(crop3)
ax0.axis('off');
ax1.axis('off');
ax2.axis('off');
```

![png]({{ site.baseurl }}/assets/ODD_Boxes/output_12_0.png)


## Anchor boxes

How do you predict bounding boxes for an image? If you understand anchor boxes then you are well on your way to understanding deep learning based object detection. 

The approach involves taking lots of boxes of different sizes and aspect ratios evenly spaced apart on a grid across the image. We call these "anchor boxes". The  you have a large enough combination of sizes and scales, then then there is a good chance that one or more of them bound the objects in the image fairly well. This turns the problem into something of a classification task whereby you try to predict what is inside each region. For example, does it have an object or not and if so what is the class label of that object? 

We can keep all those boxes which confidently predict an object and discard the rest. However the initial boxes are not good enough by themselves so we also want the model to predict a 
linear transformation of the boxes consisting of stretching and translation that till turn them into boxes that better fit the objects. 



# Anchor generation  

Let us start with a single output feature map that has a stride of $S$ with respect to the input. We also have a set of anchor boxes of different areas and aspect ratios (ARs). Let us say all these possible combinations of the areas [128^2, 256^2, 512^2] and ARs [1:2, 1:1, 2:1] which gives us 9 different boxes. These boxes are replicated across all the pixels in the feature map so that each one is responsible for finding objects in a part of the image. (It is also possible to have a strided grid where we skip pixels but we will assume for now that all the pixels are used).

These boxes will be defined with respect to the original input dimensions. Each of the pixels maps to an $S \times S$ square of the input. The centre of the pixel is at the centre of this $S x S$ box. So suppose $S=16$ which is the stride of the output of the 4 block of ResNet50 or ResNet101 with respect to the input. For instance the ImageNet imagesize of 224 x 224 the output of the fourth block is $14 x 14$. For the top-left $S \times S$ region in input $[x_1, y_1, x_2, y_2] = [0, 0, 15, 15]$ leading to centre coordinates of $[7.5, 7.5]$

To construct our grid of anchor boxes let us first split the input image into a grid of $S \times S$ non-overlapping cells each corresponding to a pixel of the output feature map. We will always assume that our image dimensions are exactly divisible by $S$ since this can been achieved by padding the image appropriately. These are then the base boxes. They will be rescaled into different sizes and aspect ratios to produce a set of concentric anchor boxes. The predictions contained in at each pixel location of the output feature map correspond to these anchor boxes (how exactly we will discuss later). 

![png]({{ site.baseurl }}/assets/ODD_Boxes/anchors12x12_resized.jpg)

# Implementing anchor generation

Another way to think about anchor generation, which is how we implement it: 

- First to consider just the top left corner $S \times S$ box in the grid. 
- Now rescale it as described above to generate a set of anchors. 

Try writing a function `generate_anchors` that does this.

<div class="collapse-generate_anchors">
<div markdown="1">
```python
def generate_anchors(
    stride=16, sizes=(128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """
    Adapted from https://github.com/facebookresearch/maskrcnn-benchmark
    """
    
    sizes = np.array(sizes)
    aspect_ratios = np.array(aspect_ratios)
    
    base_anchor = np.array([1, 1, stride, stride], dtype=np.float) - 1 # (1,)
    w, h, x_ctr, y_ctr =  to_whxy(base_anchor) # (1,)
    base_area = w * h
    
    widths = np.round(np.sqrt(base_area / aspect_ratios)) # (A,)
    heights = np.round(widths * aspect_ratios) # (A,)
    
    scales = sizes / stride
    widths = scales[None] * widths[:, None] # (A, S)
    heights = scales[None] * heights[:, None]  # (A, S)
    
    # Each is (A, S)
    x1 = x_ctr - 0.5 * (widths - 1)
    y1 = y_ctr - 0.5 * (heights - 1)
    x2 = x_ctr + 0.5 * (widths - 1)
    y2 = y_ctr + 0.5 * (heights - 1)
    
    anchors = np.stack([x1, y1, x2, y2], axis=-1) #(A, S, 4)
    
    return np.reshape(anchors, [-1, 4]) # (A*S, 4)
```
</div>
</div>

Let us plot a set of anchor boxes

```python
fig, ax = plt.subplots(1, figsize=(8,8))
anchors = generate_anchors()
for box in anchors:
    plot_rectangle(box, ax, linewidth=1);

plt.xlim(anchors[:, 0].min() - 25, anchors[:, 2].max() + 25)
plt.ylim(anchors[:, 1].min() - 25, anchors[:, 3].max() + 25);
plt.plot(*to_whxy(box)[-2:], 'cx')
ax.set_xticks([7.5])
ax.set_yticks([7.5])
plot_rectangle([0, 0, 15, 15], ax, clr='r');
ax.invert_yaxis();
ax.grid();
```
![png]({{ site.baseurl }}/assets/ODD_Boxes/output_18_0.png)

Once we have a set of anchors, we then translate them across and down the image with a step size of $S$ so that we get a set of anchors concentric with each of the cells in the image. 

<div class="collapse-grid_anchors">
<div markdown="1">
```python
def grid_anchors(base_anchors, stride, width, height, img_width, img_height):
    """
    base_anchors: (N, 4)
    H = height
    W = width
    
    Adapted from https://github.com/facebookresearch/maskrcnn-benchmark
    """
    shifts_x = np.arange(0, width * stride, stride) # (W,)
    shifts_y = np.arange(0, height * stride, stride) # (H,)
    
    # Indexing is 'xy' by default in np/tf but 'ij' in pytorch 
    # (so they reverse these in the mrcnn benchmark code)
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y) 
    
    shifts = np.stack([shifts_x, shifts_y, shifts_x, shifts_y], axis=-1) # (H, W, 4)
    
    # (H, W, 1, 4) + (1, 1, N, 4) -> (H, W, N, 4)
    anchors = shifts[..., None, :] + base_anchors[None, None]
    
    inside = ((anchors >= 0).all(axis=-1) & 
             (anchors <= [img_width-1, img_height-1, img_width-1, img_height-1]).all(axis=-1))
    
    return anchors, inside
```
</div>
</div>


<!--<div>
Let us say that we have an input image size of $H \times W$ and $A$ area/AR combinations. Thus we have a $H/S \times W/S$ grid of $S \times S$ cells, each with $A$ anchor boxes so $A\cdot H \cdot W/S^2$ anchor boxes in total. Since the anchor boxes have an area typically much larger than $S^2$, as shown in the figure above, and they overlap quite a lot resulting in good coverage across the image but also making them difficult to visualise. Instead we will visualise a set of tiny anchors on the $12 \times 12$ input illustrated earlier.


```python
sizes=(0.5, 1.2, 3)
aspect_ratios=(0.5, 1, 2)
stride = 4
grid, inside = grid_anchors(generate_anchors(stride, sizes=sizes, aspect_ratios=aspect_ratios), stride,
                            width=3, height=3,
                            img_width=12, img_height=12)
fig, ax = plt.subplots(1, figsize=(8,8))
grid_bbox = [0, 0, 11, 11]
plot_rectangle(grid_bbox, ax, clr='k', linewidth=2);
clrs = ['coral', 'blueviolet', 'dodgerblue'] * 3
for cell in grid.reshape((-1, 9, 4)):
    for i, box in enumerate(cell):
        plot_rectangle(box, ax, linewidth=1, clr=clrs[i]);
    plt.plot(*to_whxy(box)[-2:], 'cx')
    
        
plt.xlim(-2, 13)
plt.ylim(-2, 13);
ax.set_xticks(np.arange(-1.5, 13., 1));
ax.set_yticks(np.arange(-1.5, 13., 1));

ax.grid();
```


![png]({{ site.baseurl }}/assets/ODD_Boxes/output_21_0.png)


The whole image can be regarded as a bounding box with coordinates $[0, 0, H-1, W-1]$. 
The `inside` array returned by `grid_anchors` indicates whether the coordinates of each anchor box are bounded by the coordinates of the entire image bounding-box. 
Notice that some of these boxes fall outside of the image grid. How do we handle these? We could clip them so that they fall within the image boundary. We could also merely discard them. 


```python
sizes=(0.5, 1.2, 3)
aspect_ratios=(0.5, 1, 2)
stride = 4
grid, inside = grid_anchors(generate_anchors(stride, sizes=sizes, aspect_ratios=aspect_ratios), stride,
                            width=3, height=3,
                            img_width=12, img_height=12)

print('Num total boxes {}, num inside {}'.format(inside.size, inside.sum()))

clrs = ['coral', 'blueviolet', 'dodgerblue'] * 3
fig, axes = plt.subplots(1, 2, figsize=(16,8))
grid_bbox = [0, 0, 11, 11]
w, h, _, _ = to_whxy(np.array([0, 0, 11, 11]))

for ax, clip in zip(axes, [True, False]):
    plot_rectangle(grid_bbox, ax, clr='k', linewidth=2);

    for cell, isin in zip(grid.reshape((-1, 9, 4)), inside.reshape((-1, 9))):
        for i, (box, box_in) in enumerate(zip(cell, isin)):
            if not box_in:
                if clip:
                    box = np.clip(box, [0, 0, 0, 0], [w-1, h-1, w-1, h-1])
                else:
                    continue
            plot_rectangle(box, ax, linewidth=1, clr=clrs[i]);
        ax.plot(*to_whxy(box)[-2:], 'cx')
    
    ax.set_title('Boxes that cross image boundary are {}'.format(
        'clipped' if clip else 'discarded'
    ))    
    ax.set_xlim(-2, 13)
    ax.set_ylim(-2, 13);
    ax.set_xticks(np.arange(-1.5, 13., 1));
    ax.set_yticks(np.arange(-1.5, 13., 1));

    ax.grid();
```

    Num total boxes 81, num inside 69



![png]({{ site.baseurl }}/assets/ODD_Boxes/output_23_1.png)


This is the approach adopted in the [Faster-RCNN paper](https://arxiv.org/abs/1506.01497) which introduced the notion of anchor boxes:

> During training, we ignore all cross-boundary anchors so they do not contribute to the loss. 

This is reason they give for discarding them  

> If the boundary-crossing outliers are not ignored in training, they introduce large, difficult to correct error terms in the objective, and training does not converge.

However during testing they 

> [W]e clip [the boxes] to the image boundary.


Next time we will discuss how predictions and ground truth bounding boxes are defined relative to anchors.

# Deltas
Since we would like to make predictions for arbitrary sized images, instead of predicting bounding box coordinates, we predict parameters for a transform of the anchor boxes. The anchor bounding box is given as


$$[x_a, y_a, h_a, w_a]$$


For the target bounding box $[x, y, h, w]$ we generate targets "deltas" $t = [t_x, t_y, t_w, t_h]$ as follows:

$$t_x = (x - x_a) / h_a \\
t_y = (y - y_a) / w_a \\
t_w =  \log(w / w_a) \\
t_h = \log(h / h_a) $$

and the predictions $p$ are the interpreted as follows:

$$p_x = (\hat{x} - x_a) / h_a \\
p_y = (\hat{y} - y_a) / w_a \\
p_w =  \log(\hat{w} / w_a) \\
p_h = \log(\hat{h} / h_a) $$


```python
def split_coords_dims(box):
    return box[..., :2], box[..., 2:]

def box2delta(box, ref):
    box_coords, box_dims = split_coords_dims(box)
    ref_coords, ref_dims = split_coords_dims(ref)
    delta_coords = (box_coords - ref_coords) / ref_dims
    delta_dims = tf.math.log(box_dims / ref_dims)
    return tf.concat([delta_coords, delta_dims], axis=-1)


def delta2box(delta, ref):
    delta_coords, delta_dims = split_coords_dims(delta)
    ref_coords, ref_dims = split_coords_dims(ref)
    box_coords = ref_coords + (delta_coords * ref_dims) 
    box_dims = ref_dims * tf.math.exp(delta_dims)
    return tf.concat([box_coords, box_dims], axis=-1)
```


```python
b = tf.stack([[8, 12, 15., 25]])
a = tf.stack([[8, 8, 16., 16]])
```


```python
split_coords_dims(b)
```




    (<tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[ 8., 12.]], dtype=float32)>,
     <tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[15., 25.]], dtype=float32)>)




```python
split_coords_dims(a)
```




    (<tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[8., 8.]], dtype=float32)>,
     <tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[16., 16.]], dtype=float32)>)




```python
box2delta(b, a)
```




    <tf.Tensor: shape=(1, 4), dtype=float32, numpy=
    array([[ 0.        ,  0.25      , -0.06453852,  0.4462871 ]],
          dtype=float32)>




```python
delta2box(box2delta(b, a), a)
```




    <tf.Tensor: shape=(1, 4), dtype=float32, numpy=array([[ 8., 12., 15., 25.]], dtype=float32)>



# Bounding box loss

When training the predicted "deltas" $p$ are compared directly with $t$ rather than comparing the predicting bounding box $[\hat{x}, \hat{y}, \hat{w}, \hat{h}]$ with the target bounding box $[x, y, h, w]$. This enables the loss to be unaffected by the image dimensions. Typically the smooth-${L_1}$ loss is used. 


$$\text{smooth}_{L_1}(x)=\left\{
                \begin{array}{ll}
                  0.5x^2, \text{ }\text{ }\text{ }\text{ }\text{if } |x| < 1\\
                  |x| - 0.5, \text{ }\text{ }\text{ }\text{ }\text{otherwise}
                \end{array}
              \right.$$
              
This essentially proportional to the $L2$ loss between -1 and 1 and proportional to the $L1$ loss everywhere else. It avoids the abrupt change in slope around $0$ present in the $L1$ loss. On the other hand if we just used the $L_2$ loss, since the coordinates and the predictions are unbounded, it could potentially increase a lot and lead to exploding gradients.  


```python
x = np.linspace(-2, 2, 201)

L1_based = np.abs(x) - 0.5
L2_based = 0.5 * (x ** 2)
smooth_L1 = np.where(np.abs(x) < 1, L2_based, L1_based)
```


```python
plt.figure(figsize=(12, 8))
plt.plot(x, L1_based, label='L1_based', color='cornflowerblue', linewidth=3)
plt.plot(x, L2_based, label='L2_based', color='forestgreen', linewidth=3)
plt.plot(x, smooth_L1, label='smooth_L1', color='red', linestyle='--', linewidth=3)
plt.title('Loss functions', fontsize=16)
plt.legend();
```


![png]({{ site.baseurl }}/assets/ODD_Boxes/output_34_0.png) 
</div>-->

