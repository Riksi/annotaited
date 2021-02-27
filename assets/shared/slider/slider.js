var slideIndex = 1;

function plusSlides(n) {
    showSlides(slideIndex += n);
    }

    function currentSlide(n) {
    showSlides(slideIndex = n);
    }

    function showSlides(n) {
    var i;
    var slides = document.getElementsByClassName("mySlides");
    var dots = document.getElementsByClassName("dot");
    if (n > slides.length) {slideIndex = 1}    
    if (n < 1) {slideIndex = slides.length}
    for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";  
    }
    document.getElementsByClassName("prev")[0].style.display = (slideIndex==1)?"none":"initial";
    slides[slideIndex-1].style.display = "block";  
    }

function makeSlider(slider, captions, imgs, name){
    
    captions.forEach((caption, idx)=>{
        var slide = document.querySelector('#tmp-slide').content.querySelector(
            'div.mySlides'
        ).cloneNode(true);
        if (idx>0){
            slide.style.display='none';
        }
        slide.querySelector('.text').innerText = caption;
        slide.querySelector('img').src = "/active-learning/assets/" + name + '/' + imgs[idx];
        slider.appendChild(slide);
    })

    var buttons = ['prev', 'next'];
    buttons.forEach((btn)=>{
        console.log(btn);
        var link = document.createElement('a');
        link.className = btn;
        var prev = btn == 'prev';
        var inc = prev ? -1 : 1;
        var symbol = prev ? '❮': '❯';
        link.innerText = symbol;
        link.addEventListener('click', (e)=>plusSlides(inc));
        if(prev) link.style.display='none';
        slider.appendChild(link);
    })

    
    showSlides(slideIndex);
}
