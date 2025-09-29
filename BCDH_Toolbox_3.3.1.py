import os
import Metashape
import time, math
import matplotlib.pyplot as plt
import numpy as np
import json
import csv
from datetime import datetime
from PySide2 import QtCore, QtWidgets

# Import matplotlib backends correctly
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    backend_available = True
except ImportError:
    backend_available = False

# --- Globale Variablen und Hilfsfunktionen ---
store = []
last_report_data = None
REPORT_FILE = os.path.join(os.path.expanduser('~'), 'Metashape_Error_Reports.json')

# ---- Shared plotting helpers (style + gradient + mean/median) ----
def _apply_image1_style(ax):
    ax.set_facecolor('white')
    ax.grid(True, axis='y', color='#E9ECEF', linewidth=1.0, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#D0D3D8')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(colors='#4A4A4A', length=6, width=1.0, direction='out')

def _apply_compact_layout(fig):
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.98)

def _position_title_and_ylabel(ax, title_pad=8, ylabel_x=-0.09):
    t = ax.get_title()
    if t:
        ax.set_title(t, pad=title_pad, color='#2F2F2F')
    try:
        ax.yaxis.set_label_coords(ylabel_x, 0.5)
    except Exception:
        pass
    ax.xaxis.labelpad = 4
    ax.yaxis.labelpad = 4

def _colorize_hist_patches_data_span(patches, data_min, data_max,
                                     c0='#36DC6E', c1='#4093DF'):
    from matplotlib.colors import to_rgb
    c0 = np.array(to_rgb(c0))
    c1 = np.array(to_rgb(c1))
    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max <= data_min:
        for p in patches:
            p.set_facecolor(c1); p.set_edgecolor((0, 0, 0, 0)); p.set_linewidth(0.0)
        return
    rng = float(data_max - data_min)
    for p in patches:
        xc = p.get_x() + 0.5 * p.get_width()
        t = (xc - data_min) / rng
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        col = c0 * (1.0 - t) + c1 * t
        p.set_facecolor(col)
        p.set_edgecolor((0, 0, 0, 0))
        p.set_linewidth(0.0)

def _draw_mean_median(ax, data, color='#444444'):
    if not data:
        return None, None
    import numpy as _np
    data_arr = _np.asarray(list(data), dtype=float)
    med_v  = float(_np.median(data_arr))
    rms_v  = float(_np.sqrt(_np.mean(_np.square(data_arr))))
    h_med = ax.axvline(med_v, color=color, linestyle='-',  linewidth=1.8, label=f"Median: {med_v:.3f}")
    h_rms = ax.axvline(rms_v, color=color, linestyle='--', linewidth=1.8, label=f"RMS: {rms_v:.3f}")
    return h_med, h_rms

def _fixed_label_width(label: QtWidgets.QLabel, sample_text: str):
    fm = label.fontMetrics()
    try:
        w = fm.horizontalAdvance(sample_text)
    except Exception:
        w = fm.width(sample_text)
    label.setMinimumWidth(int(w * 1.05))

def getMarker(chunk, label):
    for marker in chunk.markers:
        if marker.label.upper() == label.upper():
            return marker
    return 0

def vect(a, b):
    result = Metashape.Vector([a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x])
    return result.normalized()

class profil(QtWidgets.QDialog):
    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)
        doc = Metashape.app.document
        chunk = doc.chunk
        self.setWindowTitle("Marker-Based Profil")

        # UI
        self.btnBuild  = QtWidgets.QPushButton("Build with Resolution…")
        self.btnExport = QtWidgets.QPushButton("Export with Resolution…")
        self.resTxt    = QtWidgets.QLabel("Resolution (m):")
        self.vectTxt   = QtWidgets.QLabel("Horizontal Markers:")
        self.vect2Txt  = QtWidgets.QLabel("Vertical Markers:")
        self.resInput  = QtWidgets.QLineEdit("0.001")

        self.llistV = [QtWidgets.QComboBox(), QtWidgets.QComboBox()]
        self.llistH = [QtWidgets.QComboBox(), QtWidgets.QComboBox()]

        # Fill boxes with Marker
        for marker in chunk.markers:
            for ilist in self.llistV + self.llistH:
                ilist.addItem(marker.label)
        
        # Layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)
        layout.addWidget(self.resTxt,    0, 0)
        layout.addWidget(self.resInput,  0, 1)
        layout.addWidget(self.vectTxt,   1, 0)
        layout.addWidget(self.vect2Txt,  2, 0)
        layout.addWidget(self.llistH[0], 1, 1)
        layout.addWidget(self.llistH[1], 1, 2)
        layout.addWidget(self.llistV[0], 2, 1)
        layout.addWidget(self.llistV[1], 2, 2)
        layout.addWidget(self.btnBuild,  3, 1)
        layout.addWidget(self.btnExport, 3, 2)
        self.setLayout(layout)

        self._prefill_resolution_from_gsd()
        
        self.btnBuild.clicked.connect(self._build_ortho_only)
        self.btnExport.clicked.connect(self._export_ortho)
        
        self.exec()
    
    def _prefill_resolution_from_gsd(self):
        try:
            import statistics, math
            doc = Metashape.app.document
            chunk = doc.chunk
            if chunk is None:
                return
            tp = getattr(chunk, "tie_points", None)
            if not tp or not getattr(tp, "points", None) or not getattr(tp, "projections", None):
                return

            points      = tp.points
            projections = tp.projections  
            tracks      = getattr(tp, "tracks", None)
            T           = chunk.transform.matrix

            # track_id -> point_index
            if tracks is not None:
                point_ids = [-1] * len(tracks)
            else:
                max_tid = 0
                for p in points:
                    if getattr(p, "valid", False):
                        max_tid = max(max_tid, int(p.track_id))
                point_ids = [-1] * (max_tid + 1)

            for pid, p in enumerate(points):
                if getattr(p, "valid", False):
                    tid = int(p.track_id)
                    if 0 <= tid < len(point_ids):
                        point_ids[tid] = pid

            def fpx_from_camera(cam):
                cal = getattr(getattr(cam, "sensor", None), "calibration", None)
                if cal and getattr(cal, "f", None):
                    try:
                        return float(cal.f)
                    except Exception:
                        pass
                if cal and getattr(cal, "fx", None) and getattr(cal, "fy", None):
                    return 0.5 * (float(cal.fx) + float(cal.fy))
                cal2 = getattr(cam, "calibration", None)
                if cal2 and getattr(cal2, "f", None):
                    return float(cal2.f)
                return None

            cams = [c for c in chunk.cameras if c.transform and getattr(c, "type", None) == Metashape.Camera.Type.Regular]
            cams = [c for c in cams if (c in projections)]
            if not cams:
                return

            per_cam_gsd = []
            for cam in cams:
                fpx = fpx_from_camera(cam)
                if not fpx or fpx <= 0:
                    continue
                C = T.mulp(cam.center)

                d_over_f = []
                for pr in projections[cam]:
                    tid = int(getattr(pr, "track_id", -1))
                    if tid < 0 or tid >= len(point_ids):
                        continue
                    pid = point_ids[tid]
                    if pid < 0:
                        continue
                    pt = points[pid]
                    if not getattr(pt, "valid", False):
                        continue

                    coord = pt.coord
                    try:
                        coord.size = 3  
                    except Exception:
                        pass
                    P = T.mulp(coord)
                    d = (C - P).norm()
                    d_over_f.append(d / fpx)

                if d_over_f:
                    per_cam_gsd.append(statistics.median(d_over_f))

            if not per_cam_gsd:
                return

            gsd_m = sum(per_cam_gsd) / len(per_cam_gsd)  # m/px
            txt = "{:.12f}".format(gsd_m).rstrip('0').rstrip('.')
            if txt:
                self.resInput.setText(txt)
        except Exception:
            # 
            pass

    def _mk_projection_and_res(self, chunk):
        try:
            res_x = res_y = float(self.resInput.text())
        except ValueError:
            Metashape.app.messageBox("Incorrect export resolution! Please use point delimiter.\n")
            return None

        # Validation
        if (self.llistV[0].currentIndex() == self.llistV[1].currentIndex() or
            self.llistH[0].currentIndex() == self.llistH[1].currentIndex()):
            Metashape.app.messageBox("Can't use the same marker for vector start and end.\n")
            return None

        if len(chunk.markers) < 2:
            Metashape.app.messageBox("At least 2 markers required.\n")
            return None

        T = chunk.transform.matrix

        # Horizontal direction vector
        mH1 = getMarker(chunk, self.llistH[0].currentText())
        mH2 = getMarker(chunk, self.llistH[1].currentText())
        horizontal = T.mulv(mH2.position - mH1.position).normalized()

        # Vertical direction vector
        mV1 = getMarker(chunk, self.llistV[0].currentText())
        mV2 = getMarker(chunk, self.llistV[1].currentText())
        vertical_raw = T.mulv(mV2.position - mV1.position)

        normal  = vect(vertical_raw, horizontal)
        vertical = vect(horizontal, normal)

        R = Metashape.Matrix([horizontal, vertical, -normal])
        origin = T.mulp(mV1.position)  
        X = (-1) * R * origin

        A = 0.0
        try:
            if mV1.reference and mV1.reference.location:
                A = float(mV1.reference.location.z)
        except Exception:
            A = 0.0

        horizontal.size = 4; horizontal.w = X.x
        vertical.size   = 4; vertical.w   = X.y + A
        normal.size     = 4; normal.w     = -X.z

        proj = Metashape.Matrix([
            horizontal,
            vertical,
            -normal,
            Metashape.Vector([0, 0, 0, 1])
        ])

        projection = Metashape.OrthoProjection()
        projection.type = Metashape.OrthoProjection.Type.Planar
        projection.crs = chunk.crs.geoccs if chunk.crs and chunk.crs.geoccs else chunk.crs
        projection.matrix = proj

        return projection, res_x, res_y


    def _build_ortho_only(self):
        doc = Metashape.app.document
        chunk = doc.chunk

        r = self._mk_projection_and_res(chunk)
        if not r:
            return
        projection, res_x, res_y = r

        self.btnBuild.setDisabled(True)
        try:
            chunk.buildOrthomosaic(
                surface_data=Metashape.DataSource.ModelData,
                blending_mode=Metashape.BlendingMode.MosaicBlending,
                fill_holes=True,
                projection=projection,
                resolution_x=res_x,
                resolution_y=res_y
            )

            if chunk.orthomosaic:
                chunk.orthomosaic.projection = projection

            Metashape.app.messageBox(
                "Orthomosaic has been created!"
            )
        except Exception as e:
            Metashape.app.messageBox("Build failed:\n{}".format(e))
        finally:
            self.btnBuild.setDisabled(False)

    def _export_ortho(self):
        doc = Metashape.app.document
        chunk = doc.chunk
        if not chunk.orthomosaic:
            Metashape.app.messageBox("No orthomosaic found. Please build first.")
            return


        project_path = Metashape.app.document.path
        project_folder = os.path.dirname(project_path) if project_path else ""
        path_output = Metashape.app.getSaveFileName("Choose Path to save Profil", project_folder, ".tif")
        if not path_output:
            return
        if not path_output.lower().endswith(".tif"):
            path_output += ".tif"

        try:
            res_x = res_y = float(self.resInput.text())
        except ValueError:
            Metashape.app.messageBox("Incorrect export resolution! Please use point delimiter.\n")
            return

        try:
            chunk.exportRaster(
                path=path_output,
                source_data=Metashape.DataSource.OrthomosaicData,
                resolution_x=res_x,
                resolution_y=res_y
            )
            Metashape.app.messageBox("Exported:\n{}".format(path_output))
        except Exception as e:
            Metashape.app.messageBox("Export failed:\n{}".format(e))



class exportProjectedMarker(QtWidgets.QDialog):
    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Export projected Marker from Ortho")
        doc = Metashape.app.document
        chunk = doc.chunk
        
        # UI-Elemente
        self.vectTxt = QtWidgets.QLabel("Horizontal Markers:")
        self.vect2Txt = QtWidgets.QLabel("Vertical Markers:")
        self.llistV = [QtWidgets.QComboBox(), QtWidgets.QComboBox()]
        self.llistH = [QtWidgets.QComboBox(), QtWidgets.QComboBox()]
        self.btnExport = QtWidgets.QPushButton("Export Marker Coordinates")
        
        # Combobox fill
        for marker in chunk.markers:
            for combo in self.llistV + self.llistH:
                combo.addItem(marker.label)
        
        # Layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.vectTxt, 0, 0)
        layout.addWidget(self.llistH[0], 0, 1)
        layout.addWidget(self.llistH[1], 0, 2)
        layout.addWidget(self.vect2Txt, 1, 0)
        layout.addWidget(self.llistV[0], 1, 1)
        layout.addWidget(self.llistV[1], 1, 2)
        layout.addWidget(self.btnExport, 2, 0, 1, 3)
        self.setLayout(layout)
        

        self.btnExport.clicked.connect(self.exportMarkers)
        self.exec_()

    def exportMarkers(self):
        doc = Metashape.app.document
        chunk = doc.chunk
        
        
        mH1 = getMarker(chunk, self.llistH[0].currentText())
        mH2 = getMarker(chunk, self.llistH[1].currentText())
        mV1 = getMarker(chunk, self.llistV[0].currentText())
        mV2 = getMarker(chunk, self.llistV[1].currentText())
        
        required_markers = [mH1, mH2, mV1, mV2]
        
        if any(not marker or not marker.enabled or marker.position is None for marker in required_markers):
            Metashape.app.messageBox("Activate all Markers for the export!")
            return
        
        T = chunk.transform.matrix
        

        if (self.llistV[0].currentIndex() == self.llistV[1].currentIndex() or 
            self.llistH[0].currentIndex() == self.llistH[1].currentIndex()):
            Metashape.app.messageBox("Can't use the same marker for vector start and end.")
            return
        

        horizontal = T.mulv(mH2.position - mH1.position).normalized()       
        vertical = T.mulv(mV2.position - mV1.position)
        
        

        normal = vect(vertical, horizontal)
        vertical = vect(horizontal, normal)
        R = Metashape.Matrix([horizontal, vertical, -normal])
        origin = T.mulp(mV1.position)
        X = (-1) * R * origin
        

        A = mV1.reference.location.z if mV1.reference.location else 0
        horizontal.size = 4; horizontal.w = X.x
        vertical.size = 4; vertical.w = X.y + A
        normal.size = 4; normal.w = -X.z
        proj = Metashape.Matrix([horizontal, vertical, -normal, Metashape.Vector([0, 0, 0, 1])])
        

        path = Metashape.app.getSaveFileName("Save Marker Coordinates", "", "Text files (*.txt)")
        if not path:
            return
        
        # Export
        with open(path, 'w') as f:
            f.write("Label;X;Y\n")
            for marker in chunk.markers:
                if marker.position:
                    point = T.mulp(marker.position)
                    projected = proj * Metashape.Vector([point.x, point.y, point.z, 1])
                    f.write(f"{marker.label};{projected.x};{projected.y}\n")
        
        Metashape.app.messageBox("Marker coordinates exported successfully!")
        self.accept()


def save_report_data(data):
    try:
        reports = []
        if os.path.exists(REPORT_FILE):
            with open(REPORT_FILE, 'r') as f:
                reports = json.load(f)
        data['timestamp'] = datetime.now().isoformat()
        reports.append(data)
        if len(reports) > 5:
            reports = reports[-5:]
        with open(REPORT_FILE, 'w') as f:
            json.dump(reports, f, indent=2)
        return data
    except Exception as e:
        print(f"Error saving report: {str(e)}")
        return None

def load_last_report():
    try:
        if os.path.exists(REPORT_FILE):
            with open(REPORT_FILE, 'r') as f:
                reports = json.load(f)
                return reports[-1] if reports else None
        return None
    except:
        return None

def calc_reprojection(chunk):
    global store
    store = []
    point_cloud = chunk.tie_points
    points = point_cloud.points
    npoints = len(points)
    projections = chunk.tie_points.projections
    err_sum = 0
    num = 0
    maxe = 0
    length_tracks = len(point_cloud.tracks)
    point_ids = [-1] * len(point_cloud.tracks)
    point_errors = {}
    for point_id in range(npoints):
        point_ids[points[point_id].track_id] = point_id
    for camera in chunk.cameras:
        if not camera.transform or not camera.enabled:
            continue
        for proj in projections[camera]:
            track_id = proj.track_id
            point_id = point_ids[track_id]
            if point_id < 0:
                continue
            point = points[point_id]
            if not point.valid:
                continue
            error = camera.error(point.coord, proj.coord).norm() ** 2
            px_error = math.sqrt(error)
            store.append(px_error)
            err_sum += error
            num += 1
            if point_id not in point_errors:
                point_errors[point_id] = [error]
            else:
                point_errors[point_id].append(error)
            if error > maxe:
                maxe = error
    sigma = math.sqrt(err_sum / num) if num > 0 else 0
    return (sigma, point_errors, maxe, num, npoints, length_tracks)

def error_report():
    global last_report_data
    doc = Metashape.app.document
    chunk = doc.chunk
    result = calc_reprojection(chunk)
    if result[3] == 0:
        print("No valid projections found!")
        return
    sigma, _, maxe, num, npoints, length_tracks = result
    store_sorted = sorted(store)
    store_length = len(store_sorted)
    f = Metashape.TiePoints.Filter()
    f.init(chunk, criterion=Metashape.TiePoints.Filter.ReprojectionError)
    min_kp = f.min_value
    max_kp = f.max_value
    current_report = {
        'initial_tie_points': length_tracks,
        'current_tie_points': npoints,
        'projections_calculated': num,
        'min_kp': min_kp,
        'max_kp': max_kp,
        'min_px': store_sorted[0],
        'median_px': store_sorted[store_length//2],
        'max_px': store_sorted[-1],
        'sigma_px': sigma
    }
    if last_report_data is None:
        last_report_data = load_last_report()
    diff_report = {}
    if last_report_data:
        for key in current_report:
            if key in last_report_data:
                diff_report[key] = current_report[key] - last_report_data[key]
    saved_report = save_report_data(current_report)
    if saved_report:
        last_report_data = saved_report
    def format_value(value, is_float=False):
        return f"{value:.4f}" if is_float else f"{value}"
    def format_diff(diff, is_float=False):
        if diff is None: return "N/A".rjust(15)
        if diff == 0: return "0".rjust(15)
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.4f}".rjust(15) if is_float else f"{sign}{diff}".rjust(15)
    print("\n" + "="*61)
    print("ERROR REPORT".center(61))
    print("="*61)
    print(f"{'Metric'.ljust(25)} | {'Current Value'.center(15)} | {'Difference'.rjust(15)}")
    print("-"*61)
    entries = [
        ("Initial tie points", current_report['initial_tie_points'], diff_report.get('initial_tie_points'), False),
        ("Current tie points", current_report['current_tie_points'], diff_report.get('current_tie_points'), False),
        ("Projections calculated", current_report['projections_calculated'], diff_report.get('projections_calculated'), False),
        ("Min KeyPoint error", current_report['min_kp'], diff_report.get('min_kp'), True),
        ("Max KeyPoint error", current_report['max_kp'], diff_report.get('max_kp'), True),
        ("Min pixel error", current_report['min_px'], diff_report.get('min_px'), True),
        ("Median pixel error", current_report['median_px'], diff_report.get('median_px'), True),
        ("Max pixel error", current_report['max_px'], diff_report.get('max_px'), True),
        ("Reprojection sigma", current_report['sigma_px'], diff_report.get('sigma_px'), True)
    ]
    for i, (label, value, diff, is_float) in enumerate(entries):
        if i == 2 or i == 4:
            print("-"*61)
        value_str = format_value(value, is_float).center(15)
        diff_str = format_diff(diff, is_float)
        print(f"{label.ljust(25)} | {value_str} | {diff_str}")
    print("="*61 + "\n")

# --- Histograms ---
def selection_over_histogram_pixel():
    if not backend_available:
        print("Matplotlib backend not available - interactive features disabled")
        return
    doc = Metashape.app.document
    chunk = doc.chunk
    result = calc_reprojection(chunk)
    if not store:
        print("No data available!")
        return

    # Histogram DATA = per-projection reprojection error [px] (like Chunk Info histograms)
    data = list(store)
    x_min = 0.0
    raw_x_max = float(max(data)) if len(data) > 0 else 0.0
    x_max = np.nextafter(raw_x_max, np.inf)  # include max in last bin
    d_min = float(min(data))
    d_max = float(max(data))

    # Selection LOGIC = RMS per 3D Tie Point (like Gradual Selection)
    _, point_errors, _, _, _, _ = result
    per_point_rms_px = {}
    for pid, errs_sq in point_errors.items():
        if not errs_sq:
            continue
        per_point_rms_px[pid] = float(np.sqrt(np.mean(errs_sq)))
    total_points = len(per_point_rms_px)


    try:
        parent = Metashape.app.getMainWindow()
    except AttributeError:
        parent = None
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Histogram for projection: Reprojection Error [px] - Median and RMS based on projections - Selection: Tie points")
    dialog.setMinimumSize(900, 640)
    layout = QtWidgets.QVBoxLayout()

    # Figure
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    _apply_compact_layout(fig)
    canvas = FigureCanvasQTAgg(fig)
    ax = fig.add_subplot(111)
    _apply_image1_style(ax)

    # Histogram from projections
    n, bins, patches = ax.hist(
        data, bins=255, range=(x_min, x_max), histtype='bar', linewidth=0.0
    )
    _colorize_hist_patches_data_span(patches, data_min=d_min, data_max=d_max,
                                     c0='#36DC6E', c1='#4093DF')
    ax.set_xlabel("Reprojection Error [px]", color='#4A4A4A')
    ax.set_ylabel("Projections", color='#4A4A4A')  # reflect per-projection histogram
    _position_title_and_ylabel(ax, title_pad=8, ylabel_x=-0.09)
    ax.margins(x=0)  # start exactly at 0

    # Median & RMS OF PROJECTIONS (RMS = sqrt(mean(err_px^2)))
    import numpy as _np
    _data_arr = _np.asarray(data, dtype=float)
    _median_val = float(_np.median(_data_arr)) if _data_arr.size else 0.0
    _rms_val = float(_np.sqrt(_np.mean(_data_arr**2))) if _data_arr.size else 0.0
    ax.axvline(_median_val, color='#444444', linestyle='-',  linewidth=1.8, label=f"Median: {_median_val:.3f}")
    rms_line = ax.axvline(_rms_val,    color='#444444', linestyle='--', linewidth=1.8, label=f"RMS: {_rms_val:.3f}")
    ax.legend(facecolor='white', frameon=True, edgecolor='#D0D3D8')

    # Threshold line (px)
    threshold = raw_x_max / 2 if raw_x_max > 0 else 0.0
    threshold_line = ax.axvline(x=threshold, color='r', linestyle='--')

    layout.addWidget(canvas)

    # Slider row
    slider_widget = QtWidgets.QWidget()
    slider_widget.setMaximumHeight(48)
    slider_layout = QtWidgets.QHBoxLayout(slider_widget)
    slider_layout.setContentsMargins(0,0,0,0)
    slider_label = QtWidgets.QLabel("Threshold:")
    slider_layout.addWidget(slider_label)

    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setMinimum(0); slider.setMaximum(1000); slider.setFixedHeight(20)
    slider.setValue(int(threshold / raw_x_max * 1000) if raw_x_max > 0 else 0)
    slider_layout.addWidget(slider, 1)

    value_label = QtWidgets.QLabel(f"{threshold:.3f} px")
    value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    _fixed_label_width(value_label, "000000.000 px")
    slider_layout.addWidget(value_label)

    # Counter shows Tie Points with RMS(px) > threshold
    def count_above(thr): return sum(1 for v in per_point_rms_px.values() if v > thr)
    counter_label = QtWidgets.QLabel(f"Selected: {count_above(threshold)} / {total_points}")
    counter_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    _fixed_label_width(counter_label, f"Selected: {total_points} / {total_points}")
    slider_layout.addWidget(counter_label)

    layout.addWidget(slider_widget)

    # Buttons
    button_layout = QtWidgets.QHBoxLayout()
    select_button = QtWidgets.QPushButton("Select Points > Threshold")
    select_button.clicked.connect(lambda: select_points_map(per_point_rms_px, threshold))
    button_layout.addWidget(select_button)
    deselect_button = QtWidgets.QPushButton("Deselect All")
    deselect_button.clicked.connect(deselect_all_points)
    button_layout.addWidget(deselect_button)
    close_button = QtWidgets.QPushButton("Close")
    close_button.clicked.connect(dialog.accept)
    button_layout.addWidget(close_button)
    layout.addLayout(button_layout)

    # Update threshold
    def update_threshold(value):
        nonlocal threshold
        threshold = value / 1000 * raw_x_max if raw_x_max > 0 else 0.0
        value_label.setText(f"{threshold:.3f} px")
        counter_label.setText(f"Selected: {count_above(threshold)} / {total_points}")
        threshold_line.set_xdata([threshold, threshold])
        canvas.draw_idle()
    slider.valueChanged.connect(update_threshold)

    dialog.setLayout(layout)
    dialog.exec_()

def selection_over_histogram_keypoint():
    if not backend_available:
        print("Matplotlib backend not available - interactive features disabled")
        return
    doc = Metashape.app.document
    chunk = doc.chunk
    tie_points = chunk.tie_points
    if not tie_points or not tie_points.points:
        print("No tie points!")
        return

    # Build projection-based dimensionless errors: u = err_px / keypoint_size
    track_to_point = {p.track_id: i for i, p in enumerate(tie_points.points) if p.valid}
    proj_units = []
    per_point_units_sq = {}

    for cam in chunk.cameras:
        if not cam.enabled or not cam.transform:
            continue
        try:
            projs = tie_points.projections[cam]
        except Exception:
            continue
        for proj in projs:
            tid = getattr(proj, "track_id", None)
            if tid is None or tid not in track_to_point:
                continue
            pid = track_to_point[tid]
            # keypoint size
            kp_size = getattr(proj, "scale", None)
            if kp_size is None:
                kp_size = getattr(proj, "size", None)
            try:
                kp_size = float(kp_size) if kp_size is not None else float("nan")
            except Exception:
                kp_size = float("nan")
            # reprojection error in px for this projection
            try:
                tp = tie_points.points[pid]
                proj_from_cam = cam.project(tp.coord)
                if hasattr(proj.coord, "x"):
                    kp_x, kp_y = float(proj.coord.x), float(proj.coord.y)
                else:
                    kp_x, kp_y = float(proj.coord[0]), float(proj.coord[1])
                err_px = ((float(proj_from_cam.x)-kp_x)**2 + (float(proj_from_cam.y)-kp_y)**2)**0.5
            except Exception:
                err_px = float("nan")
            if kp_size and kp_size != 0.0 and err_px == err_px:
                u = err_px / kp_size
                proj_units.append(u)
                per_point_units_sq.setdefault(pid, []).append(u*u)

    proj_units = [v for v in proj_units if v == v and v >= 0.0]
    if not proj_units:
        print("No valid projection-based values (dimensionless).")
        return

    # Selection metric: RMS per Tie Point (dimensionless)
    import numpy as _np
    f = Metashape.TiePoints.Filter()
    f.init(chunk, criterion=Metashape.TiePoints.Filter.ReprojectionError)
    error_scaled = list(f.values)
    total_points = len(error_scaled)

    # Histogram setup
    x_min = 0.0
    raw_x_max = float(max(proj_units))
    x_max = _np.nextafter(raw_x_max, _np.inf)
    d_min = float(min(proj_units))
    d_max = float(max(proj_units))

    # Dialog
    try:
        parent = Metashape.app.getMainWindow()
    except AttributeError:
        parent = None
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Histogram for projection: Reprojection Error (Reprojection Error [px] / Key Point Size) - Median and RMS based on projections - Selection: Tie Points")
    dialog.setMinimumSize(900, 640)
    layout = QtWidgets.QVBoxLayout()

    # Figure
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    _apply_compact_layout(fig)
    canvas = FigureCanvasQTAgg(fig)
    ax = fig.add_subplot(111)
    _apply_image1_style(ax)

    # Histogram from projections
    n, bins, patches = ax.hist(proj_units, bins=255, range=(x_min, x_max), histtype='bar', linewidth=0.0)
    _colorize_hist_patches_data_span(patches, data_min=d_min, data_max=d_max, c0='#36DC6E', c1='#4093DF')

    # Labels 
    ax.set_xlabel("Reprojection Error", color='#4A4A4A')
    ax.set_ylabel("Projections", color='#4A4A4A')
    _position_title_and_ylabel(ax, title_pad=6, ylabel_x=-0.09)
    ax.margins(x=0)

    # Median & RMS from projections (RMS = sqrt(mean(u^2)))
    _arr = _np.asarray(proj_units, dtype=float)
    _median_val = float(_np.median(_arr))
    _rms_val = float(_np.sqrt(_np.mean(_arr**2)))
    ax.axvline(_median_val, color='#444444', linestyle='-',  linewidth=1.8, label=f"Median: {_median_val:.3f}")
    rms_line = ax.axvline(_rms_val,    color='#444444', linestyle='--', linewidth=1.8, label=f"RMS: {_rms_val:.3f}")
    ax.legend(facecolor='white', frameon=True, edgecolor='#D0D3D8')

    # Slider (dimensionless threshold) -> selection on Tie Point RMS (dimensionless)
    threshold = raw_x_max / 2 if raw_x_max > 0 else 0.0
    threshold_line = ax.axvline(x=threshold, color='r', linestyle='--')
    layout.addWidget(canvas)

    slider_widget = QtWidgets.QWidget()
    slider_widget.setMaximumHeight(48)
    slider_layout = QtWidgets.QHBoxLayout(slider_widget)
    slider_layout.setContentsMargins(0,0,0,0)
    slider_label = QtWidgets.QLabel("Threshold:")
    slider_layout.addWidget(slider_label)

    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setMinimum(0); slider.setMaximum(1000); slider.setFixedHeight(20)
    slider.setValue(int(threshold / raw_x_max * 1000) if raw_x_max > 0 else 0)
    slider_layout.addWidget(slider, 1)

    value_label = QtWidgets.QLabel(f"{threshold:.3f}")
    value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    _fixed_label_width(value_label, "000000.000")
    slider_layout.addWidget(value_label)

    def count_above(thr): return sum(1 for v in error_scaled if v > thr)
    counter_label = QtWidgets.QLabel(f"Selected: {count_above(threshold)} / {total_points}")
    counter_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    _fixed_label_width(counter_label, f"Selected: {total_points} / {total_points}")
    slider_layout.addWidget(counter_label)
    layout.addWidget(slider_widget)

    # Buttons
    button_layout = QtWidgets.QHBoxLayout()
    select_button = QtWidgets.QPushButton("Select Points > Threshold")
    select_button.clicked.connect(lambda: select_points_kp(error_scaled, threshold))
    button_layout.addWidget(select_button)
    deselect_button = QtWidgets.QPushButton("Deselect All")
    deselect_button.clicked.connect(deselect_all_points)
    button_layout.addWidget(deselect_button)
    close_button = QtWidgets.QPushButton("Close")
    close_button.clicked.connect(dialog.accept)
    button_layout.addWidget(close_button)
    layout.addLayout(button_layout)

    def update_threshold(value):
        nonlocal threshold
        threshold = value / 1000 * raw_x_max if raw_x_max > 0 else 0.0
        value_label.setText(f"{threshold:.3f}")
        counter_label.setText(f"Selected: {count_above(threshold)} / {total_points}")
        threshold_line.set_xdata([threshold, threshold])
        canvas.draw_idle()
    slider.valueChanged.connect(update_threshold)

    dialog.setLayout(layout)
    dialog.exec_()

def plot_true_keypointsize_histogram(style='aqua_gradient', kde=False, bins='auto'):
    from PySide2 import QtWidgets, QtCore
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import numpy as np
    import matplotlib.pyplot as plt  # for colormaps if needed
    import traceback

    def _show_error(msg):
        try:
            Metashape.app.messageBox(msg)
        except Exception:
            print(msg)

    try:
        def _get_valid_track_map(tp):
            return {p.track_id: i for i, p in enumerate(tp.points) if getattr(p, "valid", False)}

        def _get_sizes_per_projection_all_valid(chunk):
            sizes = []
            tp = chunk.tie_points
            if not tp or not getattr(tp, "points", None):
                return sizes
            valid_tracks = _get_valid_track_map(tp)
            for cam in chunk.cameras:
                if not getattr(cam, "enabled", True) or not getattr(cam, "transform", None):
                    continue
                try:
                    projs = tp.projections[cam]
                except Exception:
                    continue
                cam_kp = getattr(cam, "keypoints", None)
                for pr in projs:
                    tid = getattr(pr, "track_id", None)
                    if tid is None or tid not in valid_tracks:
                        continue
                    size_val = getattr(pr, "scale", None)
                    if size_val is None:
                        size_val = getattr(pr, "size", None)
                    if size_val is None and cam_kp is not None:
                        try:
                            if hasattr(pr, "key_id"):
                                size_val = cam_kp[pr.key_id].size
                            elif hasattr(pr, "index"):
                                size_val = cam_kp[pr.index].size
                        except Exception:
                            size_val = None
                    try:
                        size_f = float(size_val)
                    except Exception:
                        continue
                    if size_f >= 0.0 and np.isfinite(size_f):
                        sizes.append(size_f)
            return sizes

        def _get_sizes_per_point_avg_valid(chunk):
            tp = chunk.tie_points
            if not tp or not getattr(tp, "points", None):
                return {}
            valid_tracks = _get_valid_track_map(tp)
            per_point = {}
            for cam in chunk.cameras:
                if not getattr(cam, "enabled", True) or not getattr(cam, "transform", None):
                    continue
                try:
                    projs = tp.projections[cam]
                except Exception:
                    continue
                cam_kp = getattr(cam, "keypoints", None)
                for pr in projs:
                    tid = getattr(pr, "track_id", None)
                    if tid is None or tid not in valid_tracks:
                        continue
                    size_val = getattr(pr, "scale", None)
                    if size_val is None:
                        size_val = getattr(pr, "size", None)
                    if size_val is None and cam_kp is not None:
                        try:
                            if hasattr(pr, "key_id"):
                                size_val = cam_kp[pr.key_id].size
                            elif hasattr(pr, "index"):
                                size_val = cam_kp[pr.index].size
                        except Exception:
                            size_val = None
                    try:
                        size_f = float(size_val)
                    except Exception:
                        continue
                    pid = valid_tracks[tid]
                    per_point.setdefault(pid, []).append(size_f)
            return {pid: (sum(v)/len(v)) for pid, v in per_point.items() if v}

        if not backend_available:
            _show_error("Matplotlib backend not available – plotting disabled.")
            return

        doc = Metashape.app.document
        chunk = doc.chunk
        if chunk is None:
            _show_error("Kein aktives Chunk gefunden!")
            return

        proj_sizes = _get_sizes_per_projection_all_valid(chunk)


        try:
            parent = Metashape.app.getMainWindow()
        except AttributeError:
            parent = None
        dialog = QtWidgets.QDialog(parent)
        dialog.setWindowTitle("Histogram for projections: Key Point Size (sigma of the Gaussian blur at the pyramid level of scales) - Mean based on projections - Selection: Tie points")
        dialog.setMinimumSize(1000, 700)
        layout = QtWidgets.QVBoxLayout(dialog)

        fig = Figure()
        fig.patch.set_facecolor('white')
        _apply_compact_layout(fig)
        canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        _apply_image1_style(ax)
        layout.addWidget(canvas)

        if not proj_sizes:
            ax.text(0.5, 0.5, "Keine gültigen Keypoint-Größen (Projektionen) gefunden.", ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            btn = QtWidgets.QPushButton("Schließen")
            btn.clicked.connect(dialog.accept)
            layout.addWidget(btn)
            dialog.setWindowModality(QtCore.Qt.ApplicationModal)
            dialog.show()
            QtWidgets.QApplication.processEvents()
            dialog.exec_()
            return

        proj_sizes = [float(s) for s in proj_sizes if np.isfinite(s) and s >= 0.0]
        d_min = float(min(proj_sizes))
        d_max_raw = float(max(proj_sizes))
        x_min = d_min
        x_max = np.nextafter(d_max_raw, np.inf)

        # Auto bins
        q75, q25 = np.percentile(proj_sizes, [75, 25])
        iqr = max(q75 - q25, 0.0)
        span = max(d_max_raw - d_min, 1e-6)
        if iqr > 0:
            bw = 2 * iqr * (len(proj_sizes) ** (-1/3))
            n_bins = int(np.clip(span / max(bw, 1e-6), 20, 400))
        else:
            n_bins = min(max(int(span), 30), 400)

        # Histogram
        counts, edges, patches = ax.hist(
            proj_sizes, bins=n_bins, range=(x_min, x_max), histtype='bar', linewidth=0.0
        )
        _colorize_hist_patches_data_span(patches, data_min=d_min, data_max=d_max_raw, c0='#36DC6E', c1='#4093DF')
        ax.margins(x=0)

        ax.set_xlabel("Key point size", color='#4A4A4A')
        ax.set_ylabel("Projections", color='#4A4A4A')
        _position_title_and_ylabel(ax, title_pad=6, ylabel_x=-0.09)

        # Mean from projections 
        mean_from_projections = float(np.mean(np.asarray(proj_sizes, dtype=float)))
        ax.axvline(mean_from_projections, color='#444444', linestyle='-', linewidth=1.8,
                   label=f"Mean: {mean_from_projections:.3f}")
        ax.legend(facecolor='white', frameon=True, edgecolor='#D0D3D8')

        # Selection: tie-point avg 
        size_map = _get_sizes_per_point_avg_valid(chunk)
        total_points = len(size_map)

        threshold = x_min + 0.5 * (d_max_raw - x_min) if d_max_raw > x_min else x_min
        threshold_line = ax.axvline(x=threshold, color='r', linestyle='--')

        slider_widget = QtWidgets.QWidget()
        slider_widget.setMaximumHeight(48)
        slider_layout = QtWidgets.QHBoxLayout(slider_widget)
        slider_layout.setContentsMargins(0,0,0,0)
        slider_label = QtWidgets.QLabel("Threshold:")
        slider_layout.addWidget(slider_label)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(0); slider.setMaximum(1000); slider.setFixedHeight(20)
        slider.setValue(int((threshold - x_min) / (d_max_raw - x_min) * 1000) if d_max_raw > x_min else 0)
        slider_layout.addWidget(slider, 1)

        value_label = QtWidgets.QLabel(f"{threshold:.3f}")
        value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        _fixed_label_width(value_label, "000000.000")
        slider_layout.addWidget(value_label)

        def count_above(thr):
            return sum(1 for v in size_map.values() if v > thr)

        counter_label = QtWidgets.QLabel(f"Selected: {count_above(threshold)} / {total_points}")
        counter_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        _fixed_label_width(counter_label, f"Selected: {total_points} / {total_points}")
        slider_layout.addWidget(counter_label)

        layout.addWidget(slider_widget)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        select_button = QtWidgets.QPushButton("Select Points > Threshold")
        select_button.clicked.connect(lambda: select_points_map(size_map, threshold))
        button_layout.addWidget(select_button)
        deselect_button = QtWidgets.QPushButton("Deselect All")
        deselect_button.clicked.connect(deselect_all_points)
        button_layout.addWidget(deselect_button)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        def update_threshold(value):
            nonlocal threshold
            if d_max_raw > x_min:
                threshold = x_min + (value/1000.0) * (d_max_raw - x_min)
            else:
                threshold = x_min
            value_label.setText(f"{threshold:.3f}")
            counter_label.setText(f"Selected: {count_above(threshold)} / {total_points}")
            threshold_line.set_xdata([threshold, threshold])
            canvas.draw_idle()

        slider.valueChanged.connect(update_threshold)

        dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        dialog.show()
        QtWidgets.QApplication.processEvents()
        dialog.exec_()

    except Exception as e:
        _show_error("Fehler in plot_true_keypointsize_histogram:\n" + str(e) + "\n" + traceback.format_exc())
        return

def selection_over_histogram_imagecount():
    if not backend_available:
        print("Matplotlib backend not available - interactive features disabled")
        return

    doc = Metashape.app.document
    chunk = doc.chunk
    tp = getattr(chunk, "tie_points", None)
    if tp is None or not getattr(tp, "points", None):
        print("No tie points!")
        return

    track_to_pid = {p.track_id: i for i, p in enumerate(tp.points) if getattr(p, "valid", False)}
    img_count_map = {}  
    for cam in chunk.cameras:
        if not getattr(cam, "enabled", True) or not getattr(cam, "transform", None):
            continue
        try:
            projs = tp.projections[cam]
        except Exception:
            continue
        for pr in projs:
            tid = getattr(pr, "track_id", None)
            if tid is None or tid not in track_to_pid:
                continue
            pid = track_to_pid[tid]
            img_count_map[pid] = img_count_map.get(pid, 0) + 1

    if not img_count_map:
        print("No valid per-point image counts found.")
        return

    counts = list(img_count_map.values())
    total_points = len(img_count_map)


    import numpy as _np
    c_min = int(min(counts))
    c_max = int(max(counts))
    left = c_min - 0.5
    right = c_max + 0.5
    if left >= right:
        left, right = c_min - 0.5, c_min + 0.5
    edges = _np.arange(left, right + 1.0, 1.0)
    x_min = left
    x_max = _np.nextafter(right, _np.inf)

    try:
        parent = Metashape.app.getMainWindow()
    except AttributeError:
        parent = None
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Histogram for tie points: Images per Tie Point – Median based on tie points – Selection: Tie Points (< threshold)")
    dialog.setMinimumSize(900, 640)
    layout = QtWidgets.QVBoxLayout(dialog)

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    _apply_compact_layout(fig)
    canvas = FigureCanvasQTAgg(fig)
    ax = fig.add_subplot(111)
    _apply_image1_style(ax)

    n, bins, patches = ax.hist(counts, bins=edges, range=(x_min, x_max), histtype='bar', linewidth=0.0)
    _colorize_hist_patches_data_span(patches, data_min=float(c_min), data_max=float(c_max), c0='#36DC6E', c1='#4093DF')

    ax.set_xlabel("Images per Tie Point", color='#4A4A4A')
    ax.set_ylabel("Tie Points", color='#4A4A4A')
    _position_title_and_ylabel(ax, title_pad=8, ylabel_x=-0.09)
    ax.margins(x=0)

    counts_arr = _np.asarray(counts, dtype=float)
    med_val = float(_np.median(counts_arr)) if counts_arr.size else 0.0
    ax.axvline(med_val, color='#444444', linestyle='-', linewidth=1.8, label=f"Median: {med_val:.3f}")
    ax.legend(facecolor='white', frameon=True, edgecolor='#D0D3D8')

    threshold = med_val
    thr_line = ax.axvline(x=threshold, color='r', linestyle='--')

    layout.addWidget(canvas)

    # Slider
    slider_widget = QtWidgets.QWidget()
    slider_widget.setMaximumHeight(48)
    slider_layout = QtWidgets.QHBoxLayout(slider_widget)
    slider_layout.setContentsMargins(0, 0, 0, 0)
    slider_label = QtWidgets.QLabel("Threshold:")
    slider_layout.addWidget(slider_label)

    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(1000)
    slider.setFixedHeight(20)

    span = float(c_max - c_min) if c_max > c_min else 1.0
    slider.setValue(int((threshold - c_min) / span * 1000.0))
    slider_layout.addWidget(slider, 1)

    value_label = QtWidgets.QLabel(f"{threshold:.3f}")
    value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    _fixed_label_width(value_label, "000000.000")
    slider_layout.addWidget(value_label)

    def count_below(thr):
        return sum(1 for v in img_count_map.values() if v < thr)

    counter_label = QtWidgets.QLabel(f"Selected: {count_below(threshold)} / {total_points}")
    counter_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    _fixed_label_width(counter_label, f"Selected: {total_points} / {total_points}")
    slider_layout.addWidget(counter_label)

    layout.addWidget(slider_widget)

    # --- Buttons ---
    button_layout = QtWidgets.QHBoxLayout()

    def select_points_less_than(thr):
        points = tp.points
        for p in points:
            p.selected = False
        sel = 0
        for pid, val in img_count_map.items():
            if pid < len(points) and val < thr:
                points[pid].selected = True
                sel += 1
        print(f"Selected {sel} tie points with image count < {thr:.3f}")
        Metashape.app.update()

    select_button = QtWidgets.QPushButton("Select Points < Threshold")
    select_button.clicked.connect(lambda: select_points_less_than(threshold))
    button_layout.addWidget(select_button)

    deselect_button = QtWidgets.QPushButton("Deselect All")
    deselect_button.clicked.connect(deselect_all_points) 
    button_layout.addWidget(deselect_button)

    close_button = QtWidgets.QPushButton("Close")
    close_button.clicked.connect(dialog.accept)
    button_layout.addWidget(close_button)

    layout.addLayout(button_layout)

    # --- Slider-Update ---
    def update_threshold(value):
        nonlocal threshold
        threshold = c_min + (value / 1000.0) * (span if span > 0 else 1.0)
        value_label.setText(f"{threshold:.3f}")
        counter_label.setText(f"Selected: {count_below(threshold)} / {total_points}")
        thr_line.set_xdata([threshold, threshold])
        canvas.draw_idle()

    slider.valueChanged.connect(update_threshold)

    dialog.setLayout(layout)
    dialog.exec_()

def select_points(max_point_errors, threshold):
    doc = Metashape.app.document
    chunk = doc.chunk
    points = chunk.tie_points.points
    for point in points:
        point.selected = False
    selected_count = 0
    for point_id, max_error in max_point_errors.items():
        if max_error > threshold and point_id < len(points):
            points[point_id].selected = True
            selected_count += 1
    print(f"Selected {selected_count} points with error > {threshold:.3f} px")
    Metashape.app.update()

def select_points_kp(error_values, threshold):
    doc = Metashape.app.document
    chunk = doc.chunk
    points = chunk.tie_points.points
    for point in points:
        point.selected = False
    selected_count = 0
    for i, error in enumerate(error_values):
        if i < len(points) and error > threshold:
            points[i].selected = True
            selected_count += 1
    print(f"Selected {selected_count} points with error > {threshold:.3f}")
    Metashape.app.update()

def select_points_map(score_map, threshold):
    doc = Metashape.app.document
    chunk = doc.chunk
    points = chunk.tie_points.points
    for p in points:
        p.selected = False
    cnt = 0
    for pid, val in score_map.items():
        if pid < len(points) and val > threshold:
            points[pid].selected = True
            cnt += 1
    print(f"Selected {cnt} points with value > {threshold:.3f}")
    Metashape.app.update()

def deselect_all_points():
    doc = Metashape.app.document
    chunk = doc.chunk
    for point in chunk.tie_points.points:
        point.selected = False
    print("Deselected all tie points")
    Metashape.app.update()

def export_projections_csv():
    import math
    doc = Metashape.app.document
    chunk = doc.chunk
    if chunk is None:
        Metashape.app.messageBox("Kein aktives Chunk gefunden!")
        return
    tie_points = chunk.tie_points
    if not tie_points or len(tie_points.points) == 0:
        Metashape.app.messageBox("Keine Tie-Points im Chunk vorhanden!")
        return
    path = Metashape.app.getSaveFileName("CSV speichern unter", "", "CSV Dateien (*.csv)")
    if not path:
        return
    float_precision = 6
    fmt = lambda v: (f"{v:.{float_precision}f}" if isinstance(v, float) else str(v))
    track_to_point = {}
    for point_id, point in enumerate(tie_points.points):
        if point.valid:
            track_to_point[point.track_id] = point_id
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            "ID","Cameralabel","Keypoint ID","Keypoint X","Keypoint Y",
            "Reprojection Error","Reprojection Error [px]","Keypoint size",
            "Tie-point X","Tie-point Y","Tie-point Z"
        ])
        id_counter = 1
        exported = 0
        for camera in chunk.cameras:
            if not camera.enabled or not camera.transform:
                continue
            cam_label = camera.label if camera.label else camera.key
            projs = tie_points.projections[camera]
            for keypoint_id, proj in enumerate(projs):
                track_id = proj.track_id
                if track_id is None or track_id not in track_to_point:
                    continue
                point_id = track_to_point[track_id]
                tp = tie_points.points[point_id]
                if not tp.valid:
                    continue
                try:
                    kp_x = float(proj.coord.x) if hasattr(proj.coord, "x") else float(proj.coord[0])
                    kp_y = float(proj.coord.y) if hasattr(proj.coord, "y") else float(proj.coord[1])
                except Exception:
                    kp_x = float(proj.coord[0]) if len(proj.coord) > 0 else 0.0
                    kp_y = float(proj.coord[1]) if len(proj.coord) > 1 else 0.0
                kp_size = getattr(proj, "scale", None)
                if kp_size is None:
                    kp_size = getattr(proj, "size", 0)
                try:
                    kp_size = float(kp_size)
                except Exception:
                    kp_size = 0.0
                try:
                    tp_x = float(tp.coord.x) if hasattr(tp.coord, "x") else float(tp.coord[0])
                    tp_y = float(tp.coord.y) if hasattr(tp.coord, "y") else float(tp.coord[1])
                    tp_z = float(tp.coord.z) if hasattr(tp.coord, "z") else float(tp.coord[2])
                except Exception:
                    tp_x, tp_y, tp_z = 0.0, 0.0, 0.0
                try:
                    proj_from_cam = camera.project(tp.coord)
                    proj_cam_x = float(proj_from_cam.x) if hasattr(proj_from_cam, "x") else float(proj_from_cam[0])
                    proj_cam_y = float(proj_from_cam.y) if hasattr(proj_from_cam, "y") else float(proj_from_cam[1])
                    err_px = math.hypot(proj_cam_x - kp_x, proj_cam_y - kp_y)
                except Exception:
                    err_px = float("nan")
                if kp_size and kp_size != 0 and not math.isnan(err_px):
                    err_units = err_px / kp_size
                else:
                    err_units = float("nan")
                err_units_str = (f"{err_units:.{float_precision}f}" if not math.isnan(err_units) else "")
                err_px_str = (f"{err_px:.{float_precision}f}" if not math.isnan(err_px) else "")
                kp_size_str = (f"{kp_size:.{float_precision}f}" if isinstance(kp_size, (float,int)) else str(kp_size))
                row = [
                    id_counter, cam_label, keypoint_id,
                    f"{kp_x:.{float_precision}f}", f"{kp_y:.{float_precision}f}",
                    f"{err_units_str}", f"{err_px_str}", kp_size_str,
                    f"{tp_x:.{float_precision}f}", f"{tp_y:.{float_precision}f}", f"{tp_z:.{float_precision}f}"
                ]
                writer.writerow(row)
                id_counter += 1
                exported += 1
    Metashape.app.messageBox(f"CSV Export successfully! {exported} Projections exported.")

def build_profile():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    profil(parent)

def export_markers():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    exportProjectedMarker(parent)


# --- Menu ---
Metashape.app.addMenuItem("BCDH Toolbox/Analysis/Error Report...", error_report)
Metashape.app.addMenuItem("BCDH Toolbox/Analysis/Histogram Reprojection Error...", selection_over_histogram_keypoint)
Metashape.app.addMenuItem("BCDH Toolbox/Analysis/Histogram Reprojection Error [px]...", selection_over_histogram_pixel)
Metashape.app.addMenuItem("BCDH Toolbox/Analysis/Histogram Key Point Size (Projection accuracy)...", plot_true_keypointsize_histogram)
Metashape.app.addMenuItem("BCDH Toolbox/Analysis/Histogram Image count...", selection_over_histogram_imagecount)
Metashape.app.addMenuItem("BCDH Toolbox/Analysis/Export Projections as CSV", export_projections_csv)
Metashape.app.addMenuItem("BCDH Toolbox/Profile/Build and Export Marker-Based Profile...", build_profile)
Metashape.app.addMenuItem("BCDH Toolbox/Profile/Export projected Marker from Ortho...", export_markers)