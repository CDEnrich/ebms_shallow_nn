%% SGD with unbalanced transport for KL EBM 
% All expectations computed by reweigthing empirical averages over uniform data
% Data set size fixed.
% Training done over minibatches regenerated at every SGD step
% Unabalanced transport available.

%% input parameters
% clear all;
d1 = 3;                                         % input dimension (needs to be 3 right now for the graphics)
lamb0 = 1e-4;                                    % regularization strength 
vid1 = 1;                                       % set to 1 for video output, 0 otherwise
hc = 1;                                         % time step for weights c
hz = 1;                                         % time step for features z (set to zero for lazy training)
hw = 1;                                         % time step for weight (unbalanced transport): set to 0 for balanced transport
n_plot = 5;

n_t = 2;                                        % number of teachers
n_unit = 2^6;                                   % number of students
n_batch = 1e4;                                  % initial batch size
n_batch_max  = 1e6;                             % maximal batch size
n_data = 1e6;                                   % data size

Nt = 4e3;                                       % number of step of training

KL1 = zeros(Nt,1);
FD1 = zeros(Nt,1);

if vid1 == 1
    writerObj = VideoWriter('KLfeatunbal1.mp4','MPEG-4');
    writerObj.FrameRate = 4;
    open(writerObj);
end

%% energy function
f1 = @(x,z,c) mean(max(x'*z,0).*(repmat(c,size(x,2),1)),2);

%% teacher
ct = -2.5*ones(1,n_t);
% zt = [0;1;0];
% zt = randn(d1,n_t);
% zt(:,1) = [1;.8;0];
% zt(:,2) = [-1;.8;0];
zt = [-0.4907,  0.4253;0.7621, -0.3558;0.4224, -0.8321];
zt = zt./sqrt(repmat(sum(zt.^2,1),d1,1));

%% data set creation
xt = randn(d1,n_data);
xt = xt./sqrt(repmat(sum(xt.^2,1),d1,1));
Zt = mean(exp(-f1(xt,zt,ct)));
lZt = log(Zt);

%% initial weigths and features 
z = randn(d1,n_unit);
z = z./sqrt(repmat(sum(z.^2,1),d1,1));
c = zeros(1,n_unit);
w = ones(1,n_unit);
z0 = z; c0 = c;                                 % save initial values
wc = w.*c;

%% graphical output
n1 = 1e2;
theta1 = linspace(0,pi,n1);
phi1 = linspace(0,2*pi,n1);
[theta,phi] = meshgrid(theta1,phi1);
x2 = sin(theta).*cos(phi);
y2 = sin(theta).*sin(phi);
z2 = cos(theta);
f2 = 0*z2;
for k = 1:n_t
    f2 = f2+ct(k)*max(x2*zt(1,k)+y2*zt(2,k)+z2*zt(3,k),0)/n_t;
end
f2 = f2 - min(f2(:));

figure(1);clf;
subplot(1,2,1)
s = surf(x2,y2,z2,exp(-f2));
s.EdgeColor = 'none';
colorbar
hold on
axis equal
axis off
xlim([-1.3,1.3])
ylim([-1.3,1.3])
zlim([-1.3,1.3])
hold on
for k=1:n_t
    plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
end
view([180,-20])
set(gca,'FontSize',16);
cm = max(abs(c));
for j = 1:n_unit
    cs = 1+abs(c(j))/cm;
    if c(j)>0
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[1,.5,0], 'Linewidth',2)
    else 
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[0,.5,1], 'Linewidth',2)
    end
end
title(['Iteration ',num2str(0)])
drawnow
f3 = 0*z2;
for k =1:n_unit
    f3 = f3 + wc(k)*max(x2*z(1,k)+y2*z(2,k)+z2*z(3,k),0);
end
f3 = f3/n_unit;
f3 = f3 - min(f3(:));
f3(1,1) = max(f2(:));
drawnow
subplot(1,2,2)
s = surf(x2,y2,z2,exp(-f3));
s.EdgeColor = 'none';
colorbar
hold on
axis equal
axis off
xlim([-1.3,1.3])
ylim([-1.3,1.3])
zlim([-1.3,1.3])
hold on
for k=1:n_t
    plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
end
view([180,-20])
set(gca,'FontSize',16);
cm = max(abs(wc));
for j = 1:n_unit
    cs = 1+abs(wc(j))/cm;
    if wc(j)>0
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[1,.5,0], 'Linewidth',2)
    else 
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[0,.5,1], 'Linewidth',2)
    end
end
title(['Iteration ',num2str(0)])
drawnow
if vid1 == 1
    frame = getframe(gcf);
    writeVideo(writerObj,frame);
end

%%  training    
% lamb = logspace(-1,log(lamb0),Nt);
lamb = lamb0*ones(1,Nt);%linspace(1e-1,lamb0,Nt);
ehc = exp(-lamb*hc);                                % exponential integrator for c if lambda nonzero
if lamb == 0
    iehc = hc;
else
    iehc = (1-ehc)./lamb;
end

for i = 1:Nt
    if i > 2000
        n_plot = 500;
    elseif i > 100
        n_plot = 1e2;
    elseif i > 40
        n_plot = 2e1;
    end
%     if i>Nt/2; lamb = 1e-3; ehc = exp(-lamb*hc); iehc = (1-ehc)/lamb; end
    % create batch
    lZbe = log(mean(exp(-f1(xt,z,wc))));
%     tt = exp(f1m+lZbe);
%     if tt < 1/2; n_batch = 2*n_batch; fprintf('Batch size increased by factor 2\n'); end
%     if n_batch > n_batch_max; break; end
    xb = randn(d1,n_batch);
    xb = xb./sqrt(repmat(sum(xb.^2,1),d1,1));
    lZb = log(mean(exp(-f1(xb,z,wc))));
    KL1(i) = lZbe-lZt+mean((f1(xt,z,wc)-f1(xt,zt,ct)).*(exp(-f1(xt,zt,ct)-lZt)),1);
    FD1(i) = mean(sum((z*((z'*xt>0).*repmat(wc',1,size(xt,2)))/n_unit ...
        - zt*((zt'*xt>0).*repmat(ct',1,size(xt,2),1))/n_t).^2,1)'.*exp(-f1(xt,zt,ct)-lZt));
    % SGD
    dc = mean(max(xt'*z,0).*repmat(exp(-f1(xt,zt,ct)-lZt),1,n_unit),1) ...
        - mean(max(xb'*z,0).*repmat(exp(-f1(xb,z,wc)-lZb),1,n_unit),1);
    dz = (xt*((xt'*z>0).*(exp(-f1(xt,zt,ct)-lZt)*(sign(c).*min(abs(wc),1)))))/n_data ...
        - (xb*((xb'*z>0).*(exp(-f1(xb,z,wc)-lZb)*(sign(c).*min(abs(wc),1)))))/n_batch;
    dw = c.*dc;
    dz = dz - repmat(sum(z.*dz,1),d1,1).*z;
    c = c*ehc(i) - dc*iehc(i);
    z = z - hz*dz;
    z = z./sqrt(repmat(sum(z.^2,1),d1,1));
    w = w.*exp(-hw*dw);
    w = w/sum(w)*n_unit;
    wc = w.*c;
    
    if mod(i,n_plot)==0
        figure(1);clf;
        subplot(1,2,1)
        s = surf(x2,y2,z2,exp(-f2));
        s.EdgeColor = 'none';
        colorbar
        hold on
        axis equal
        axis off
        xlim([-1.3,1.3])
        ylim([-1.3,1.3])
        zlim([-1.3,1.3])
        hold on
        for k=1:n_t
            plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
        end
        view([180,-20])
        set(gca,'FontSize',16);
        cm = max(abs(wc));
        for j = 1:n_unit
            cs = 1+abs(wc(j))/cm;
            if c(j)>0
                plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[1,.5,0], 'Linewidth',2)
            else
                plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[0,.5,1], 'Linewidth',2)
            end
        end
        title(['Iteration ',num2str(i)])
        drawnow
        f3 = 0*z2;
        for k =1:n_unit
            f3 = f3 + wc(k)*max(x2*z(1,k)+y2*z(2,k)+z2*z(3,k),0);
        end
        f3 = f3/n_unit;
        f3 = f3 - min(f3(:));
        f3(1,1) = max(f2(:));
        f2(1,1) = min(f3(:));
        subplot(1,2,2)
        s = surf(x2,y2,z2,exp(-f3));
        s.EdgeColor = 'none';
        colorbar
        hold on
        axis equal
        axis off
        xlim([-1.3,1.3])
        ylim([-1.3,1.3])
        zlim([-1.3,1.3])
        hold on
        for k=1:n_t
            plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
        end
        view([180,-20])
        set(gca,'FontSize',16);
        cm = max(abs(wc));
        for j = 1:n_unit
            cs = 1+abs(wc(j))/cm;
            if c(j)>0
                plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[1,.5,0], 'Linewidth',2)
            else
                plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[0,.5,1], 'Linewidth',2)
            end
        end
        title(['Iteration ',num2str(i)])
        drawnow
        if vid1 == 1
            frame = getframe(gcf); 
            writeVideo(writerObj,frame); 
        end
    end
end

if vid1 == 1; close(writerObj); end
%% KL
figure(2);clf;
plot(KL1);
hold on
grid on
xlabel('SGD iteration','FontSize',16,'FontAngle','italic');
ylabel('KL div','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',16);

figure(3);clf;
loglog(KL1);
hold on
grid on
xlabel('SGD iteration','FontSize',16,'FontAngle','italic');
ylabel('KL div','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',16);

%%
figure(4);clf;
plot(FD1);
hold on
grid on
xlabel('SGD iteration','FontSize',16,'FontAngle','italic');
ylabel('Fisher div','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',16);

figure(5);clf;
loglog(FD1);
hold on
grid on
xlabel('SGD iteration','FontSize',16,'FontAngle','italic');
ylabel('Fisher div','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',16);

%%
f20 = 0*z2;
for k = 1:n_t
    f20 = f20+ct(k)*max(x2*zt(1,k)+y2*zt(2,k)+z2*zt(3,k),0)/n_t;
end
figure(6);clf;
subplot(1,2,1)
s = surf(x2,y2,z2,f20);
s.EdgeColor = 'none';
colorbar
hold on
axis equal
axis off
xlim([-1.3,1.3])
ylim([-1.3,1.3])
zlim([-1.3,1.3])
hold on
for k=1:n_t
    plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
end
view([120,-20])
set(gca,'FontSize',16);
cm = max(abs(wc));
for j = 1:n_unit
    cs = 1+abs(wc(j))/cm;
    if c(j)>0
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[1,.5,0], 'Linewidth',2)
    else 
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[0,.5,1], 'Linewidth',2)
    end
end
title(['Final energy '])
drawnow
f3 = 0*z2;
for k =1:n_unit
    f3 = f3 + wc(k)*max(x2*z(1,k)+y2*z(2,k)+z2*z(3,k),0);
end
f3 = f3/n_unit;
drawnow
subplot(1,2,2)
s = surf(x2,y2,z2,f3);
s.EdgeColor = 'none';
colorbar
hold on
axis equal
axis off
xlim([-1.3,1.3])
ylim([-1.3,1.3])
zlim([-1.3,1.3])
hold on
for k=1:n_t
    plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
end
view([120,-20])
set(gca,'FontSize',16);
cm = max(abs(wc));
for j = 1:n_unit
    cs = 1+abs(wc(j))/cm;
    if c(j)>0
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[1,.5,0], 'Linewidth',2)
    else 
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[0,.5,1], 'Linewidth',2)
    end
end
title(['Final energy'])
drawnow

