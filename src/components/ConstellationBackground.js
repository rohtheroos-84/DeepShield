import React, { useRef, useEffect } from 'react';

const ConstellationBackground = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let width = 1;
    let height = 1;
    const dpr = () => window.devicePixelRatio || 1;

    const resizeCanvas = () => {
      const vw = window.visualViewport?.width || window.innerWidth;
      const vh = window.visualViewport?.height || window.innerHeight;
      const scaleX = vw / width;
      const scaleY = vh / height;

      canvas.style.width = `${vw}px`;
      canvas.style.height = `${vh}px`;
      canvas.width = Math.round(vw * dpr());
      canvas.height = Math.round(vh * dpr());
      ctx.setTransform(dpr(), 0, 0, dpr(), 0, 0);

      width = vw;
      height = vh;

      if (stars.length > 0) {
        stars.forEach((star) => {
          star.x = Math.min(Math.max(star.x * scaleX, 0), width);
          star.y = Math.min(Math.max(star.y * scaleY, 0), height);
        });
      }
    };

    const stars = [];
    const starCount = 150;

    resizeCanvas();
    
    for (let i = 0; i < starCount; i++) {
      stars.push({
        x: Math.random() * width,
        y: Math.random() * height,
        radius: Math.random() * 1.5 + 0.5,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5
      });
    }
    
    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = 'white';
      stars.forEach(star => {
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
        ctx.fill();
      });
      for (let i = 0; i < starCount; i++) {
        for (let j = i + 1; j < starCount; j++) {
          const dx = stars[i].x - stars[j].x;
          const dy = stars[i].y - stars[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 100) {
            const opacity = 1 - dist / 100;
            ctx.strokeStyle = `rgba(255,255,255,${opacity})`;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(stars[i].x, stars[i].y);
            ctx.lineTo(stars[j].x, stars[j].y);
            ctx.stroke();
          }
        }
      }
    };

    const update = () => {
      stars.forEach(star => {
        star.x += star.vx;
        star.y += star.vy;
        if (star.x < 0 || star.x > width) star.vx *= -1;
        if (star.y < 0 || star.y > height) star.vy *= -1;
      });
    };

    const animate = () => {
      update();
      draw();
      requestAnimationFrame(animate);
    };

    animate();

    const handleResize = () => resizeCanvas();
    window.addEventListener('resize', handleResize);
    window.visualViewport?.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      window.visualViewport?.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: -1,
        backgroundColor: '#121212'
      }}
    />
  );
};

export default ConstellationBackground;
