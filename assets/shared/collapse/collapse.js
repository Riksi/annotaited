    window.onload = function(){
      $("div[class^=collapse]").each(
        function(i, elem){
          var p = document.createElement('p');
          var btn = document.createElement('button');
          btn.className = elem.className;
          btn.innerText = "Show/Hide";
          btn.onclick = () => {$('div.' + elem.className).toggle();};
          p.appendChild(btn)
          elem.parentNode.insertBefore(p, elem);
        }
      )
      $('#toggle-active').click(function(){
        if (this.className == 'active'){
          this.className = 'inactive';
          $("div[class^=collapse]").each(
            (idx, i)=>{i.style.display = 'block'});
          $("button[class^=collapse]").each(
            (idx, i)=>{i.style.display = 'none'});
        }
        else{
          this.className = 'active';
          $("div[class^=collapse]").each(
            (idx, i)=>{i.style.display = 'none'});
          $("button[class^=collapse]").each(
            (idx, i)=>{i.style.display = 'block'});
        }
      })

      var resizeBtn = function(){
        if ($(window).scrollTop() >= 50) { 
          $("#toggle-active").css({"display":"block"})
        }
        else{
          $("#toggle-active").css({"display":"none"})
          // $("#toggle-active").css({
          //   "right":(window.matchMedia("(min-width: 600px)").matches)?"0px":"70px",
          //  "top":"-2px"})
        }
      }
      
      $(window).on('scroll', resizeBtn);

      //window.matchMedia("(min-width: 600px)").addListener(resizeBtn);

    }