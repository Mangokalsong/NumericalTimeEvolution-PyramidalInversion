import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from os.path import dirname, join
from scipy.constants import hbar

class SchroedingerTimeEvolution:
    def __init__(self, Delta=1e-3, deltatau=1e-4, runtime = 300, sigma_0=0.04, sigma_0_pyr=6.439e-12, k_0=20, pyr_inv_distance = 1/8, x_0=38.15e-12, m = 2.47*1.66e-27):
        self.Delta = Delta                              # position step size
        self.deltatau = deltatau                        # time step size
        self.T_k = int(runtime/self.deltatau * 1e-4)    # number of time steps   # runtime is the total time of the simulation in dimensionless units (tau)
        self.N = int(1/Delta)                           # number of position steps
        self.z_n = np.linspace(-1/2, 1/2, self.N)       # position array
        self.sigma_0 = sigma_0                          
        self.k_0 = k_0                                  
        self.pyr_inv_distance = pyr_inv_distance        # the normalized distance between nitrogen atom and plane of hydrogen atoms during 
                                                        # pyramidal inversion. Determines the width of the wave function and its position in
                                                        # the infinite square well potential it is placed.
        self.sigma_0_pyr = sigma_0_pyr
        self.x_0 = x_0                                  # position of nitrogen atom during pyramidal inversion
        self.m = m                                      # =3m_N * m_H / (3m_N + m_H) where m_N is the mass of nitrogen atom and m_H is the mass of hydrogen atom                            
        self.a = self.x_0/self.pyr_inv_distance         # normalization factor for pyramidal inversion

    def psi_numerical_x(self, z):
        """Wave function of a free particle in an infinite square well potential at t=0."""
        return np.exp((-z**2)/(4*self.sigma_0**2) + 1j*self.k_0*z)
    
    def psi_analytical_x_t(self, z, tau=0):
        """Analytical solution to the time dependent Schroedinger equation for a free particle in an infinite square well potential."""
        b = np.sqrt(self.sigma_0**2 + 1j*tau/2)
        return 1/b * np.exp(-z**2/(4*b**2))
    
    def psi_analytical_alt_x_t(self, z, tau=0):         # found online, corresponds to the numerical solution
        """
        Alternate analytical solution to the time dependent Schroedinger equation for a free particle in an infinite square well potential.
        This solution has initial velocity equal to that of psi_numerical_x, and is therefore more comparable to the numerical solution.
        Found online at https://physicspages.com/pdf/Quantum%20mechanics/Free%20particle%20as%20moving%20Gaussian%20wave%20packet.pdf
        """
        b = np.sqrt(1 + 1j*tau/(2*self.sigma_0**2))
        return 1/b * np.exp(-z**2/(4*self.sigma_0**2*b**2)) * np.exp((4*1j*self.sigma_0**2*self.k_0*z-1j*2*self.sigma_0**2*self.k_0**2*tau) / (4*self.sigma_0**2*b**2))

    def psi_pyramidal_inversion_x(self, z):
        """Unitless wave function of nitrogen atom in ammonia molecule (NH3) at time t=0. Used to show pyramidal inversion"""
        return np.exp(-(z-self.x_0/self.a)**2 / (4*(self.sigma_0_pyr/self.a)**2))

    def v_infsquare(self, z):
        """The time independent potential energy for a free particle in an infinite square well potential."""
        if abs(z) > 1/2:
            return float("inf")
        else:
            return 0
    
    def v_pyramidal_inversion(self, z, omega_0=3.033e14):
        """The time independent potential energy of a nitrogen atom in the ammonia molecule (NH3). Used to show pyramidal inversion."""
        return (self.a**4 * self.m**2 * omega_0**2) / (8*hbar**2 * (self.x_0/self.a)**2) * (z**2 - (self.x_0/self.a)**2)**2

    def tau_to_t(self, tau):
        """Converts unitless time tau to seconds."""
        return self.a**2 * self.m * tau / hbar
        
    def Crank_Nicholson_matrix(self):
        h_hat = np.zeros((self.N, self.N))
        i, j = np.indices(h_hat.shape)
        h_hat[i == j] = 1/self.Delta**2
        h_hat[i == j+1] = -1/(2*self.Delta**2)
        h_hat[i == j-1] = -1/(2*self.Delta**2)
        h_hat += np.diag(self.v_k)
        Crank_minus = np.eye(self.N) - 1j*self.deltatau*h_hat
        Crank_plus = np.linalg.inv(np.eye(self.N) + 1j*self.deltatau*h_hat)
        return np.matmul(Crank_plus, Crank_minus)

    def numerical_schroedinger_time_evolution(self, psi_func=psi_numerical_x, v_func=v_infsquare, normalize_every_step=True):
        """
        Calculates the numerical time evolution of a wave function from tau=0 to tau=runtime.
        Outputs a 2D array of shape (T_k, N) where T_k is the number of time steps and N is the number of position steps.
        """
        print('calculating numerical time evolution: 0%', end='\r')
        psi_k = np.array([psi_func(z) for z in self.z_n])
        psi_k[0] = 0
        psi_k[-1] = 0
        normalization_factor = 1/np.sqrt(np.sum(abs(psi_k)**2)*self.Delta)
        psi_k = psi_k * normalization_factor
        self.v_k = np.array([v_func(z) for z in self.z_n])
        M = self.Crank_Nicholson_matrix()
        psi_k_t = np.zeros((self.T_k, self.N), dtype=complex)
        psi_k_t[0] = psi_k

        if normalize_every_step == True:        # in case wave function is not intially normalized
            for i in range(1, self.T_k):
                psi_k_t[i] = np.matmul(M, psi_k_t[i-1])
                normalization_factor = 1/np.sqrt(np.sum(abs(psi_k_t[i])**2)*self.Delta)
                psi_k_t[i] = psi_k_t[i] * normalization_factor

                if i / self.T_k * 100 % 5 == 0:
                    print(f'calculating numerical time evolution: {i/self.T_k*100:.0f}%', end='\r')
        else:
            for i in range(1, self.T_k):
                psi_k_t[i] = np.matmul(M, psi_k_t[i-1])
                
                if i / self.T_k * 100 % 5 == 0:
                    print(f'calculating time evolution: {i/self.T_k*100:.0f}%', end='\r')
        print(f'calculating numerical time evolution: 100%')
        return psi_k_t

    def analytical_schroedinger_time_evolution(self, psi_func=psi_analytical_alt_x_t):
        """
        Calculates the analytical time evolution of a wave function from tau=0 to tau=runtime, given that the wave function is function of time and position.
        Outputs a 2D array of shape (T_k, N) where T_k is the number of time steps and N is the number of position steps.
        """
        print('calculating analytical time evolution: 0%', end='\r')
        psi_k = np.array([psi_func(z) for z in self.z_n])
        normalization_factor = 1/np.sqrt(np.sum(abs(psi_k)**2)*self.Delta)
        psi_k = psi_k * normalization_factor
        psi_k_t = np.zeros((self.T_k, self.N), dtype=complex)
        psi_k_t[0] = psi_k
        for i in range(1, self.T_k):
            psi_k_t[i] = np.array([psi_func(z, tau=i*self.deltatau*2) for z in self.z_n]) * normalization_factor
            if i / self.T_k * 100 % 5 == 0:
                print(f'calculating analytical time evolution: {i/self.T_k*100:.0f}%', end='\r')
        print(f'calculating analytical time evolution: 100%')
        return psi_k_t
    
    def animate_distribution(self, psi_k_t_1, psi_k_t_2 = None, double_plot=False, save_gif=False, save_mp4=False, interval=30, plot_type='pdf', plot_every_nth_step=1):
        """
        Plots the time evolution of one or two wave functions in position space as an animation.
        plot_type='pdf' plots the probability density function of the wave function.
        plot_type='real_vs_imaginary' plots the real and imaginary parts of the wave function.
        plot_type='pdf_pyr_inv' plots the probability density function of the wave function during pyramidal inversion.
        double_plot=True plots two wave functions in the same plot.
        plot_every_nth_step=1 plots every step, plot_every_nth_step=2 plots every second step, etc.
        interval=30 is the time between frames in milliseconds.
        Saving as gif/mp4 file is saved in the same directory. FFmpeg is required to save as mp4.
        """
        def save_gif_function(filename):
            current_dir = dirname(__file__)
            file_path = join(current_dir, filename + ".gif")
            anim.save(file_path, fps=60)
        
        def save_mp4_function(filename):
            current_dir = dirname(__file__)
            file_path = join(current_dir, filename + ".mp4")
            writer = FFMpegWriter(metadata=dict(artist='Me'), bitrate=1800, fps=60)
            anim.save(file_path, writer=writer)

        if plot_type == 'pdf':
            fig, ax = plt.subplots()
            ax.set_xlim(-1/2, 1/2)
            ax.set_ylim(0, max(abs(psi_k_t_1[0]))**2 + 0.1)
            ax.set_xlabel('z')
            ax.set_ylabel(r'$|\psi(z)|^2$')
            ax.set_title('Animation of wavefunction pdf in position space over time')
            if double_plot == True:
                line1, = ax.plot([], [], lw=2, color='red', label='pdf_numerical')
                line2, = ax.plot([], [], lw=2, color='blue', label='pdf_analytical')
            else:
                line3, = ax.plot([], [], lw=2, color='purple', label='pdf', marker='o', markersize=4)
            progress = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='square', facecolor='grey', alpha=0.2))
            ax.legend()

            if double_plot == True:
                def init():
                    line1.set_data([], [])
                    line2.set_data([], []) 
                    progress.set_text('')
                    return line1, line2, progress,
            
                def animate(i):
                    x = self.z_n
                    y1 = abs(psi_k_t_1[i])**2
                    y2 = abs(psi_k_t_2[i])**2

                    line1.set_data(x, y1)
                    line2.set_data(x, y2)
                    progress.set_text(fr'$\tau$ = {i*self.deltatau*1e4:.0f} ' r'$\cdot10^{-4}$')
                    return line1, line2, progress,
                anim = FuncAnimation(fig, animate, init_func=init, frames=self.T_k, interval=interval, blit=True)

                if save_gif == True:
                    print('saving gif...')
                    save_gif_function(filename='pdf_t')
                if save_mp4 == True:
                    print('saving mp4...')
                    save_mp4_function(filename='pdf_t')
                plt.show()
                
            else:
                def init():
                    line3.set_data([], [])
                    progress.set_text('')
                    return line3, progress,

                def animate(i):
                    x = self.z_n
                    y3 = abs(psi_k_t_1[i])**2

                    line3.set_data(x, y3)
                    progress.set_text(fr'$\tau$ = {i*self.deltatau*1e4:.0f} ' r'$\cdot10^{-4}$' f'\nintegral={np.sum(y3)*self.Delta:.3f}')
                    return line3, progress,
                anim = FuncAnimation(fig, animate, init_func=init, frames=self.T_k, interval=interval, blit=True)

                if save_gif == True:
                    print('saving gif...')
                    save_gif_function(filename='pdf_t')
                if save_mp4 == True:
                    print('saving mp4...')
                    save_mp4_function(filename='pdf_t')
                plt.show()
            
        if plot_type == 'real_vs_imaginary':
            fig, ax = plt.subplots()
            ax.set_xlim(-1/2, 1/2)
            ax.set_ylim(-max(abs(psi_k_t_1[0]))**2, max(abs(psi_k_t_1[0]))**2)
            ax.set_xlabel('z')
            ax.set_ylabel(r'$\psi(z)$')
            ax.set_title('Animation of imaginary and real parts of wavefunction in position space over time')
            progress = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='square', facecolor='grey', alpha=0.2))

            if double_plot == True:
                print("Double plot not implemented for imaginary vs real plot")
            else:
                line1, = ax.plot([], [], lw=2, color='red', label='psi_real')
                line2, = ax.plot([], [], lw=2, color='blue', label='psi_imaginary')
                ax.legend()

                def init():
                    line1.set_data([], [])
                    line2.set_data([], []) 
                    progress.set_text('')
                    return line1, line2, progress,
            
                def animate(i):
                    x = self.z_n
                    y1 = psi_k_t_1[i].real
                    y2 = psi_k_t_1[i].imag

                    line1.set_data(x, y1)
                    line2.set_data(x, y2)
                    progress.set_text(fr'$\tau$ = {i*self.deltatau*1e4:.0f} ' r'$\cdot10^{-4}$')
                    return line1, line2, progress,
                anim = FuncAnimation(fig, animate, init_func=init, frames=self.T_k, interval=interval, blit=True)

                if save_gif == True:
                    print('saving gif...')
                    save_gif_function(filename='psi_real_vs_imaginary')
                if save_mp4 == True:
                    print('saving mp4...')
                    save_mp4_function(filename='psi_real_vs_imaginary')
                plt.show()

        if plot_type == 'pdf_pyr_inv':
            fig, ax = plt.subplots()
            ax.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
            ax.set_xlim(-1/2, 1/2)
            ax.set_ylim(0, max(abs(psi_k_t_1[0]))**2 + 0.1)
            ax.set_xlabel('z')
            ax.set_ylabel(r'$|\psi(z)|^2$')
            ax.set_title(r'pdf of nitrogen nucleus during pyramidal inversion of $NH_3$')
            line1, = ax.plot([], [], lw=2, color='darkorange', label='pdf')
            vertical_line1, = ax.plot([], [], lw=2, color='darkblue', linestyle='--')
            vertical_line2, = ax.plot([], [], lw=2, color='darkblue', label=r'$\pm z_0$', linestyle='--')
            progress = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='square', facecolor='grey', alpha=0.2))
            ax.legend()

            #plot vertical lines
            vert1 = self.pyr_inv_distance
            vert2 = -self.pyr_inv_distance
            vert_range = [0, max(abs(psi_k_t_1[0]))**2 + 0.1]

            def init():
                line1.set_data([], []) 
                vertical_line1.set_data([], [])
                vertical_line2.set_data([], [])
                progress.set_text('')
                return line1, vertical_line1, vertical_line2, progress,
            
            psi_k_t_1 = psi_k_t_1[::plot_every_nth_step]

            def animate(i):
                x = self.z_n
                y1 = abs(psi_k_t_1[i])**2
                y2 = vert1
                y3 = vert2

                line1.set_data(x, y1)
                vertical_line1.set_data(y2, vert_range)
                vertical_line2.set_data(y3, vert_range)
                progress.set_text(fr'$\tau$ = {i*plot_every_nth_step*self.deltatau*1e4:.0f} ' r'$\cdot10^{-4}$' '\n' fr'$t =$ {i*plot_every_nth_step*self.tau_to_t(self.deltatau)*1e12:.0f} ps' )
                return line1, vertical_line1, vertical_line2, progress,
            anim = FuncAnimation(fig, animate, init_func=init, frames=self.T_k//plot_every_nth_step, interval=interval, blit=True)

            if save_gif == True:
                print('saving gif...')
                save_gif_function(filename='pdf_pyr_inv')
            if save_mp4 == True:
                print('saving mp4...')
                save_mp4_function(filename='pdf_pyr_inv')
            plt.show()

# if __name__ == "__main__":

#     S = SchroedingerTimeEvolution(Delta=1e-3, deltatau=1e-4, runtime=400)
#     psi_k_t_analytical = S.analytical_schroedinger_time_evolution(psi_func=S.psi_analytical_alt_x_t)
#     psi_k_t_numerical = S.numerical_schroedinger_time_evolution(psi_func=S.psi_numerical_x, v_func=S.v_infsquare)
#     S.animate_distribution(psi_k_t_numerical, psi_k_t_analytical, double_plot=True, interval=30, save_gif=True, plot_type='pdf')

#     pyr_inv = SchroedingerTimeEvolution(Delta=1e-2, deltatau=1e-3, runtime=850000, pyr_inv_distance = 1/8)
#     psi_k_t_pyr_inv = pyr_inv.numerical_schroedinger_time_evolution(psi_func=pyr_inv.psi_pyramidal_inversion_x, v_func=pyr_inv.v_pyramidal_inversion, normalize_every_step=True)
#     pyr_inv.animate_distribution(psi_k_t_pyr_inv, interval=1, save_gif=False, save_mp4=False, plot_type='pdf_pyr_inv', plot_every_nth_step=60)