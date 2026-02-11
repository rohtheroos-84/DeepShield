import { useEffect, useRef } from "react";
import { Renderer, Camera, Geometry, Program, Mesh } from "ogl";
import "./Particles.css";

const Particles = () => {
  const containerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Renderer
    const renderer = new Renderer({ alpha: true, antialias: true });
    const gl = renderer.gl;
    container.appendChild(gl.canvas);
    gl.clearColor(0, 0, 0, 0);

    // Camera
    const camera = new Camera(gl, { fov: 45 });
    camera.position.set(0, 0, 20);

    // Resize Handler
    const resize = () => {
      renderer.setSize(container.clientWidth, container.clientHeight);
      camera.perspective({ aspect: gl.canvas.width / gl.canvas.height });
    };
    window.addEventListener("resize", resize);
    resize();

    // Mouse Movement Variables
    let mouseX = 0, mouseY = 0, targetX = 0, targetY = 0;

    // Mouse Move Listener (Same Direction)
    const handleMouseMove = (event) => {
      const { innerWidth, innerHeight } = window;
      mouseX = (event.clientX / innerWidth) * 2 - 1;
      mouseY = -(event.clientY / innerHeight) * 2 + 1;
    };
    window.addEventListener("mousemove", handleMouseMove);

    // Particle Data
    const numParticles = 1000;
    const positions = new Float32Array(numParticles * 3);
    const velocities = new Float32Array(numParticles * 3);
    const depths = new Float32Array(numParticles);

    for (let i = 0; i < numParticles; i++) {
      if (Math.random() < 0.4) {
        positions[i * 3] = (Math.random() < 0.5 ? -1 : 1) * (10 + Math.random() * 20);
      } else {
        positions[i * 3] = (Math.random() - 0.5) * 40;
      }

      positions[i * 3 + 1] = (Math.random() - 0.5) * 30;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 30;

      depths[i] = positions[i * 3 + 2] / 30;

      velocities[i * 3] = (Math.random() - 0.5) * 0.005;
      velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.005;
      velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.005;
    }

    // Geometry
    const geometry = new Geometry(gl, {
      position: { size: 3, data: positions },
      depth: { size: 1, data: depths },
    });

    // Shader Program
    const program = new Program(gl, {
      vertex: `
        attribute vec3 position;
        attribute float depth;
        uniform mat4 projectionMatrix, viewMatrix;
        uniform float time;
        uniform vec2 mouse;
        varying float vDepth;

        void main() {
          vec3 animatedPosition = position;

          // Floating Effect (Smooth)
          animatedPosition.x += sin(time * 0.1 + position.y * 0.1) * 0.2;
          animatedPosition.y += cos(time * 0.1 + position.x * 0.1) * 0.2;
          animatedPosition.z += sin(time * 0.05 + position.z * 0.1) * 0.1;

          // Cursor Effect (SAME DIRECTION as cursor movement)
          animatedPosition.x += mouse.x * 1.5; 
          animatedPosition.y += mouse.y * 1.5;

          vDepth = depth;
          gl_PointSize = 4.5 + (depth * 3.0); // Larger size for visibility
          gl_Position = projectionMatrix * viewMatrix * vec4(animatedPosition, 1.0);
        }
      `,
      fragment: `
        precision highp float;
        varying float vDepth;

        void main() {
          vec2 uv = gl_PointCoord - 0.5;
          float alpha = 1.0 - smoothstep(0.3, 0.5, length(uv));

          // **PURE WHITE PARTICLES**
          float brightness = 1.5 + vDepth * 0.8;  // Increased brightness
          float depthFade = 0.8 + vDepth * 0.5;  // Improved glow intensity
          gl_FragColor = vec4(1.2 * brightness, 1.2 * brightness, 1.2 * brightness, alpha * depthFade);
        }
      `,
      transparent: true,
      uniforms: {
        time: { value: 0 },
        mouse: { value: [0, 0] },
      },
    });

    // Mesh
    const particles = new Mesh(gl, { mode: gl.POINTS, geometry, program });

    let time = 0;
    const update = () => {
      time += 0.01;
      program.uniforms.time.value = time;

      // Smooth Mouse Movement (Slow Transition)
      targetX += (mouseX - targetX) * 0.1;
      targetY += (mouseY - targetY) * 0.1;
      program.uniforms.mouse.value = [targetX, targetY];

      // Update Particle Positions
      for (let i = 0; i < numParticles; i++) {
        positions[i * 3] += velocities[i * 3];
        positions[i * 3 + 1] += velocities[i * 3 + 1];
        positions[i * 3 + 2] += velocities[i * 3 + 2];

        if (Math.abs(positions[i * 3]) > 25) velocities[i * 3] *= -1;
        if (Math.abs(positions[i * 3 + 1]) > 15) velocities[i * 3 + 1] *= -1;
        if (Math.abs(positions[i * 3 + 2]) > 15) velocities[i * 3 + 2] *= -1;
      }

      geometry.attributes.position.needsUpdate = true;

      renderer.render({ scene: particles, camera });
      animationId = requestAnimationFrame(update);
    };

    let animationId = requestAnimationFrame(update);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMouseMove);
      if (container.contains(gl.canvas)) {
        container.removeChild(gl.canvas);
      }
    };
  }, []);

  return <div ref={containerRef} className="particles-container" />;
};

export default Particles;
